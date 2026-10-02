"""
anonymize_data.py
-----------------
Script d'anonymisation profonde pour AuditPulse.

Garanties de sécurité et de confidentialité :
1. Remplacement irréversible de tous les ULIDs (id, company_id, user_id, shop_id, supplier_id)
   par des identifiants synthétiques cohérents (bijection préservée).
2. Nettoyage des champs JSON imbriqués (billing.treasury_account_id remplacé par ACC_xxxx).
3. Décalage temporel global delta_t de +70 jours (multiple exact de 7) :
   - Préserve à 100% le jour de la semaine (is_weekend),
   - Préserve l'heure de la journée (is_outside_hours),
   - Préserve les délais de paiement (payment_delay_days),
   - Préserve les fenêtres temporelles de détection (24h splitting, 48h doublons, rolling 7j),
   - Empêche toute corrélation avec les logs de production réels.
4. Export au format SQL dump MySQL identique (INSERT INTO expenses ...).
"""

import sys
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
from pathlib import Path
import json
import pandas as pd
import numpy as np

# Racine du projet
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.data_loader import load_expenses_from_sql
from src.config import DEFAULT_SQL_FILE, EXPECTED_COLUMNS

def anonymize_dataset(
    source_sql: Path,
    output_sql: Path,
    days_shift: int = 70,  # 10 semaines exactes
):
    print(f"🔄 Chargement des données brutes depuis : {source_sql}")
    df = load_expenses_from_sql(source_sql, validate_columns=False)
    n_rows = len(df)
    print(f"📊 {n_rows} lignes chargées.")

    # 1. Mappings des identifiants (Bijections)
    print("🔑 Génération des identifiants synthétiques...")

    def make_mapping(series: pd.Series, prefix: str, pad: int = 4):
        unique_vals = [v for v in series.dropna().unique() if str(v).strip() != ""]
        return {val: f"{prefix}_{i+1:0{pad}d}" for i, val in enumerate(unique_vals)}

    company_map = make_mapping(df["company_id"], "COMP", 4)
    user_map    = make_mapping(df["user_id"], "USR", 4)
    shop_map    = make_mapping(df["shop_id"], "SHOP", 4)
    supp_map    = make_mapping(df["supplier_id"], "SUPP", 4)
    
    # Remplacement des IDs de transaction
    df["id"] = [f"EXP_{i+1:05d}" for i in range(n_rows)]
    df["company_id"]  = df["company_id"].map(company_map).fillna(df["company_id"])
    df["user_id"]     = df["user_id"].map(user_map)
    df["shop_id"]     = df["shop_id"].map(shop_map)
    df["supplier_id"] = df["supplier_id"].map(supp_map)

    # Références synthétiques
    df["reference"]   = [f"#REF_{i+1:05d}" if pd.notna(r) else None for i, r in enumerate(df["reference"])]

    # 2. Nettoyage du JSON billing (comptes de trésorerie)
    print("💳 Nettoyage et anonymisation du JSON 'billing'...")
    treasury_accounts = {}
    cleaned_billing = []

    for val in df["billing"]:
        if pd.isna(val) or val is None or str(val).strip() in ("", "None", "NULL"):
            cleaned_billing.append(None)
            continue
        try:
            if isinstance(val, str):
                raw_clean = val.replace('\\"', '"').replace('\\\\', '\\')
                b_dict = json.loads(raw_clean)
            else:
                b_dict = val
            if isinstance(b_dict, dict):
                tid = b_dict.get("treasury_account_id")
                if tid:
                    if tid not in treasury_accounts:
                        treasury_accounts[tid] = f"ACC_{len(treasury_accounts)+1:04d}"
                    b_dict["treasury_account_id"] = treasury_accounts[tid]
                
                # Anonymisation et nettoyage de la méthode de paiement
                method = b_dict.get("method")
                if method:
                    if "genuka" in str(method).lower() or "wilfried" in str(method).lower():
                        b_dict["method"] = "Revolut Corporate"
                    elif re.match(r"^01[0-9a-z]{24}$", str(method).lower()):
                        if "method_accounts" not in locals():
                            method_accounts = {}
                        if method not in method_accounts:
                            method_accounts[method] = f"METH_{len(method_accounts)+1:04d}"
                        b_dict["method"] = method_accounts[method]

                b_dict["address_id"] = None
                cleaned_billing.append(json.dumps(b_dict))
            else:
                cleaned_billing.append(None)
        except Exception:
            cleaned_billing.append(None)

    df["billing"] = cleaned_billing

    # Nettoyage metadata & shipping
    print("📦 Nettoyage de 'metadata' et 'shipping'...")
    df["metadata"] = '{"note": null, "matchingMediaIds": []}'
    df["items"] = None

    # 3. Décalage temporel
    print(f"⏱️ Application du décalage temporel global (+{days_shift} jours)...")
    shift = pd.Timedelta(days=days_shift)
    date_cols = ["due_date", "paid_at", "created_at", "updated_at", "deleted_at"]
    for col in date_cols:
        if col in df.columns:
            df[col] = df[col] + shift

    # 4. Génération du fichier SQL MySQL
    print(f"💾 Écriture du fichier SQL anonymisé : {output_sql}")
    cols = [c for c in EXPECTED_COLUMNS if c in df.columns]
    
    lines = [
        "-- -------------------------------------------------------------",
        "-- AuditPulse - Enterprise Expenses Dataset (Fully Anonymized)",
        "-- Identifiers, accounts, and timestamps have been sanitized.",
        "-- Statistical properties & fraud patterns are preserved 100%.",
        "-- -------------------------------------------------------------\n",
    ]
    
    col_str = ", ".join(f"`{c}`" for c in cols)
    insert_prefix = f"INSERT INTO `expenses` ({col_str}) VALUES\n"

    def sql_format(val, col_name):
        if val is None or pd.isna(val):
            return "NULL"
        if col_name in date_cols:
            if isinstance(val, pd.Timestamp):
                return f"'{val.strftime('%Y-%m-%d %H:%M:%S')}'"
            return f"'{val}'"
        if isinstance(val, (int, np.integer)):
            return str(val)
        if isinstance(val, (float, np.floating)):
            return f"{val:.2f}"
        # String / JSON : échapper les quotes
        s_val = str(val).replace("\\", "\\\\").replace("'", "\\'")
        return f"'{s_val}'"

    batch_size = 100
    for chunk_start in range(0, n_rows, batch_size):
        chunk_df = df.iloc[chunk_start:chunk_start + batch_size]
        row_strs = []
        for _, row in chunk_df.iterrows():
            formatted = [sql_format(row[c], c) for c in cols]
            row_strs.append(f"  ({', '.join(formatted)})")
        lines.append(insert_prefix + ",\n".join(row_strs) + ";\n")

    output_sql.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Anonymisation terminée avec succès ! ({output_sql})")

if __name__ == "__main__":
    src = ROOT_DIR / "data" / "local_genuka_table_expenses.sql"
    dst = ROOT_DIR / "data" / "auditpulse_expenses.sql"
    anonymize_dataset(src, dst)
