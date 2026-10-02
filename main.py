"""
main.py
-------
Point d'entrée unifié en ligne de commande (CLI) pour AuditPulse.

Commandes disponibles :
    - audit     : Exécute le pipeline complet d'audit sur un dump SQL
    - api       : Lance le serveur API REST FastAPI (Uvicorn)
    - dashboard : Lance l'interface d'investigation Streamlit
    - anonymize : Anonymise un dump SQL brut de production
    - train     : Entraîne et valide les modèles ML dans le registre central

Exemples d'utilisation :
    python main.py audit
    python main.py audit --input data/auditpulse_expenses.sql --top 15 --output reports/audit_results.csv
    python main.py api --port 8000 --reload
    python main.py dashboard --port 8501
    python main.py anonymize --input raw.sql --output data/auditpulse_expenses.sql
"""

import sys
import os
from pathlib import Path
import argparse
import time

# Reconfiguration de la console Windows en UTF-8
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Racine du projet
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import DEFAULT_SQL_FILE, FINAL_ANOMALY_THRESHOLD
from src.utils.logger import get_logger

logger = get_logger("cli")

BANNER = r"""
================================================================================
   █████╗ ██╗   ██╗██████╗ ██╗████████╗██████╗ ██╗   ██╗██╗     ███████╗
  ██╔══██╗██║   ██║██╔══██╗██║╚══██╔══╝██╔══██╗██║   ██║██║     ██╔════╝
  ███████║██║   ██║██║  ██║██║   ██║   ██████╔╝██║   ██║██║     ███████╗
  ██╔══██║██║   ██║██║  ██║██║   ██║   ██╔═══╝ ██║   ██║██║     ╚════██║
  ██║  ██║╚██████╔╝██████╔╝██║   ██║   ██║     ╚██████╔╝███████╗███████║
  ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝   ╚═╝   ╚═╝      ╚═════╝ ╚══════╝╚══════╝
           Audit & Fraud Intelligence System — v1.0.0
================================================================================
"""


# ─────────────────────────────────────────────────────────────────────────────
# COMMANDE : AUDIT / SCORING
# ─────────────────────────────────────────────────────────────────────────────

def cmd_audit(args):
    """Exécute le pipeline d'audit et affiche la synthèse financière."""
    print(BANNER)
    input_file = Path(args.input) if args.input else DEFAULT_SQL_FILE
    if not input_file.exists():
        print(f"❌ Erreur : Fichier SQL introuvable : {input_file}")
        sys.exit(1)

    print(f"📂 Fichier source ciblé : {input_file}")
    print(f"⚙️  Seuil de décision   : {args.threshold}\n")
    print("⏳ Exécution de l'audit multidimensionnel (Parsing, Features, ML, Règles, Fraude)...")

    start_time = time.time()

    from src.data_loader import load_expenses_from_sql
    from src.scorer import score_anomalies

    # Chargement et scoring
    df_raw = load_expenses_from_sql(input_file)
    results = score_anomalies(df_raw)

    elapsed = time.time() - start_time

    # Métriques
    total_tx       = len(results)
    total_amount   = results["amount"].sum()
    anomalies      = results[results["is_anomaly"] == 1]
    n_anomalies    = len(anomalies)
    anomaly_rate   = (n_anomalies / total_tx * 100) if total_tx > 0 else 0
    risk_amount    = anomalies["amount"].sum()
    risk_share     = (risk_amount / total_amount * 100) if total_amount > 0 else 0
    mean_score     = results["score_final"].mean()
    mean_anom_score = anomalies["score_final"].mean() if n_anomalies > 0 else 0

    print("\n" + "=" * 78)
    print("                      SYNTHÈSE DÉCISIONNELLE D'AUDIT")
    print("=" * 78)
    print(f"  • Durée de traitement           : {elapsed:.2f} secondes")
    print(f"  • Volume total audité           : {total_tx:,} transactions")
    print(f"  • Masse financière globale      : {total_amount:,.2f} XAF")
    print("-" * 78)
    print(f"  • Transactions critiques à risque: {n_anomalies:,} ({anomaly_rate:.2f}%)")
    print(f"  • Exposition financière capturée: {risk_amount:,.2f} XAF ({risk_share:.2f}%)")
    print(f"  • Score de risque moyen (global): {mean_score:.4f} / 1.0000")
    print(f"  • Score moyen (anomalies)       : {mean_anom_score:.4f} / 1.0000")
    print("=" * 78)

    # Top N Anomalies
    top_n = args.top
    if top_n > 0 and n_anomalies > 0:
        top_df = results.nlargest(top_n, "score_final")
        print(f"\n📋 TOP {min(top_n, len(top_df))} DES TRANSACTIONS LES PLUS CRITIQUES :")
        print("-" * 105)
        print(f"{'Réf.':<10} | {'Montant (XAF)':<18} | {'Catégorie':<16} | {'Canal':<10} | {'User':<10} | {'Score':<8} | {'Motifs'}")
        print("-" * 105)
        for _, row in top_df.iterrows():
            amt_str = f"{row['amount']:,.0f} XAF"
            rules_str = row.get("rules_triggered", "")
            if not rules_str and row.get("is_fraud_pattern", 0) == 1:
                rules_str = "Pattern Fraude Caractérisé"
            # Troncature du motif si trop long
            if len(rules_str) > 35:
                rules_str = rules_str[:32] + "..."
            print(
                f"{row['id']:<10} | {amt_str:<18} | {str(row.get('expense_type', '')):<16} | "
                f"{str(row.get('source', '')):<10} | {str(row.get('user_id', '')):<10} | "
                f"{row['score_final']*100:>5.1f}% | {rules_str}"
            )
        print("=" * 105)

    # Export optionnel
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".parquet":
            results.to_parquet(out_path, index=False)
        elif out_path.suffix.lower() == ".json":
            results.to_json(out_path, orient="records", date_format="iso", indent=2)
        else:
            results.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\n💾 Rapport d'audit complet exporté avec succès dans : {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# COMMANDE : API REST FASTAPI
# ─────────────────────────────────────────────────────────────────────────────

def cmd_api(args):
    """Démarre le serveur FastAPI."""
    print(BANNER)
    import uvicorn
    print(f"🚀 Lancement de l'API AuditPulse sur http://{args.host}:{args.port}")
    print(f"📖 Documentation interactive Swagger : http://{args.host}:{args.port}/docs")
    print(f"📘 Documentation ReDoc              : http://{args.host}:{args.port}/redoc\n")
    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload)


# ─────────────────────────────────────────────────────────────────────────────
# COMMANDE : DASHBOARD STREAMLIT
# ─────────────────────────────────────────────────────────────────────────────

def cmd_dashboard(args):
    """Démarre l'interface d'investigation Streamlit."""
    print(BANNER)
    import subprocess
    app_path = ROOT_DIR / "dashboard" / "app.py"
    if not app_path.exists():
        print(f"❌ Erreur : Dashboard introuvable à l'emplacement : {app_path}")
        sys.exit(1)

    print(f"🖥️  Démarrage du Dashboard AuditPulse sur le port {args.port}...")
    cmd = [
        sys.executable,
        "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(args.port),
        "--server.headless", "false",
    ]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Arrêt du Dashboard.")


# ─────────────────────────────────────────────────────────────────────────────
# COMMANDE : ANONYMISE
# ─────────────────────────────────────────────────────────────────────────────

def cmd_anonymize(args):
    """Anonymise un dump SQL brut."""
    print(BANNER)
    from scripts.anonymize_data import anonymize_dataset
    in_file = Path(args.input)
    if not in_file.exists():
        print(f"❌ Erreur : Fichier brut introuvable : {in_file}")
        sys.exit(1)

    out_file = Path(args.output) if args.output else DEFAULT_SQL_FILE
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"🔒 Protocole d'anonymisation industrielle AuditPulse")
    print(f"  • Source brute : {in_file}")
    print(f"  • Cible        : {out_file}")
    print(f"  • Décalage temporel : +{args.shift_days} jours (préservation des cycles)\n")

    anonymize_dataset(source_sql=in_file, output_sql=out_file, days_shift=args.shift_days)
    print(f"\n✅ Anonymisation achevée avec succès : {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
# COMMANDE : TRAIN / MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

def cmd_train(args):
    """Initialise le registre et entraîne les modèles sur les données historiques."""
    print(BANNER)
    from src.model_registry import registry
    in_file = Path(args.input) if args.input else DEFAULT_SQL_FILE
    if not in_file.exists():
        print(f"❌ Erreur : Fichier SQL introuvable : {in_file}")
        sys.exit(1)

    print(f"🔧 Entraînement et persistance des modèles dans le registre...")
    registry.initialize(in_file)
    print("\n✅ Tous les modèles (Isolation Forest, LOF, DBSCAN, Règles, Fraude) sont opérationnels !")


# ─────────────────────────────────────────────────────────────────────────────
# PARSER PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="auditpulse",
        description="AuditPulse — Moteur d'Audit Financier et Détection de Fraude par Machine Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")

    # 1. Audit
    p_audit = subparsers.add_parser("audit", help="Exécuter l'audit complet sur un dump SQL")
    p_audit.add_argument("-i", "--input", help="Chemin vers le fichier SQL à auditer (défaut : données AuditPulse)", default=None)
    p_audit.add_argument("-o", "--output", help="Chemin d'export des résultats (.csv, .parquet, .json)", default=None)
    p_audit.add_argument("-n", "--top", help="Nombre d'anomalies à afficher dans le Top (défaut : 10)", type=int, default=10)
    p_audit.add_argument("-t", "--threshold", help=f"Seuil de décision final (défaut : {FINAL_ANOMALY_THRESHOLD})", type=float, default=FINAL_ANOMALY_THRESHOLD)
    p_audit.set_defaults(func=cmd_audit)

    # 2. API
    p_api = subparsers.add_parser("api", help="Démarrer le serveur API REST FastAPI")
    p_api.add_argument("--host", default="0.0.0.0", help="Adresse IP d'écoute (défaut : 0.0.0.0)")
    p_api.add_argument("--port", type=int, default=8000, help="Port d'écoute (défaut : 8000)")
    p_api.add_argument("--reload", action="store_true", help="Activer le rechargement automatique en dev")
    p_api.set_defaults(func=cmd_api)

    # 3. Dashboard
    p_dash = subparsers.add_parser("dashboard", help="Lancer le dashboard interactif Streamlit")
    p_dash.add_argument("--port", type=int, default=8501, help="Port d'écoute (défaut : 8501)")
    p_dash.set_defaults(func=cmd_dashboard)

    # 4. Anonymize
    p_anon = subparsers.add_parser("anonymize", help="Anonymiser un dump SQL brut")
    p_anon.add_argument("-i", "--input", required=True, help="Chemin vers le dump SQL brut")
    p_anon.add_argument("-o", "--output", help=f"Chemin de sortie (défaut : {DEFAULT_SQL_FILE})", default=None)
    p_anon.add_argument("-s", "--shift-days", type=int, default=70, help="Décalage temporel en jours (défaut : 70)")
    p_anon.set_defaults(func=cmd_anonymize)

    # 5. Train
    p_train = subparsers.add_parser("train", help="Entraîner et vérifier les modèles du registre")
    p_train.add_argument("-i", "--input", help="Chemin vers le fichier SQL d'entraînement", default=None)
    p_train.set_defaults(func=cmd_train)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
