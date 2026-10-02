"""
data_loader.py
--------------
Responsabilité unique : lire le fichier SQL et retourner un DataFrame pandas propre.

Le fichier SQL contient des blocs INSERT INTO `expenses` (...) VALUES (...);
On parse ces blocs avec une regex robuste plutôt qu'un vrai parser SQL
pour rester léger et sans dépendance lourde.

Workflow :
    1. Lire le fichier SQL ligne par ligne
    2. Extraire les noms de colonnes depuis la première instruction INSERT
    3. Extraire toutes les lignes VALUES avec une regex
    4. Parser chaque ligne de valeurs en tenant compte des JSON imbriqués
    5. Construire le DataFrame, caster les types de base
    6. Retourner le DataFrame brut (le preprocessing est dans preprocessing.py)
"""

import re
import json
import pandas as pd
from pathlib import Path
from typing import Optional

from src.config import DEFAULT_SQL_FILE, EXPECTED_COLUMNS, DATE_COLUMNS
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# REGEX
# ─────────────────────────────────────────────────────────────────────────────

# Capture les noms de colonnes (avec ou sans backticks) après INSERT INTO `expenses`
RE_COLUMNS = re.compile(
    r"INSERT INTO [`]?expenses[`]?\s*\(([^)]+)\)",
    re.IGNORECASE,
)

# Capture chaque bloc VALUES (...) — on utilise une approche caractère par caractère
# pour les JSON imbriqués (voir _split_values_block)
RE_VALUES_BLOCK = re.compile(
    r"VALUES\s*(\(.*?\))\s*[,;]",
    re.IGNORECASE | re.DOTALL,
)

# Capture un bloc INSERT complet (colonnes + toutes ses VALUES)
RE_INSERT_BLOCK = re.compile(
    r"INSERT INTO [`]?expenses[`]?\s*\([^)]+\)\s*VALUES\s*(.*?);",
    re.IGNORECASE | re.DOTALL,
)


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS INTERNES
# ─────────────────────────────────────────────────────────────────────────────

def _extract_column_names(sql_text: str) -> list[str]:
    """
    Extrait les noms de colonnes du premier INSERT INTO trouvé.

    Args:
        sql_text : Contenu brut du fichier SQL.

    Returns:
        Liste de noms de colonnes (sans backticks, sans espaces).

    Raises:
        ValueError : Si aucun INSERT INTO `expenses` n'est trouvé.
    """
    match = RE_COLUMNS.search(sql_text)
    if not match:
        raise ValueError(
            "Impossible de trouver la liste de colonnes dans le SQL. "
            "Vérifiez que le fichier contient bien un INSERT INTO `expenses` (...)."
        )

    raw_cols = match.group(1)
    # Nettoie : retire les backticks, espaces, retours à la ligne
    columns = [
        col.strip().strip("`")
        for col in raw_cols.split(",")
        if col.strip()
    ]
    logger.info(f"✅ {len(columns)} colonnes extraites : {columns}")
    return columns


def _split_row_values(row_str: str) -> list[str]:
    """
    Découpe une chaîne représentant une ligne de VALUES en tenant compte :
    - des chaînes entre guillemets simples (pouvant contenir des virgules)
    - des objets JSON entre accolades (pouvant contenir des virgules et guillemets)
    - des valeurs NULL

    Args:
        row_str : Chaîne de la forme "val1, 'val2', NULL, '{\"key\": \"val\"}', ..."

    Returns:
        Liste de valeurs sous forme de chaînes brutes.
    """
    values = []
    current = []
    depth_paren = 0   # profondeur des parenthèses (pas utilisé ici mais utile en extension)
    depth_json = 0    # profondeur des accolades JSON
    in_string = False
    escape_next = False

    for char in row_str:
        if escape_next:
            current.append(char)
            escape_next = False
            continue

        if char == "\\" and in_string:
            # Caractère d'échappement MySQL
            current.append(char)
            escape_next = True
            continue

        if char == "'" and depth_json == 0:
            in_string = not in_string
            current.append(char)
            continue

        if not in_string:
            if char == "{":
                depth_json += 1
            elif char == "}":
                depth_json -= 1
            elif char == "," and depth_json == 0:
                values.append("".join(current).strip())
                current = []
                continue

        current.append(char)

    # Dernier élément
    if current:
        values.append("".join(current).strip())

    return values


def _parse_value(raw: str):
    """
    Convertit une valeur brute SQL en type Python approprié.

    - NULL           → None
    - 'texte'        → str (sans guillemets)
    - '{"key": ...}' → dict (JSON parsé)
    - 42 / 3.14      → int / float
    - Sinon          → str brute

    Args:
        raw : Valeur brute sous forme de chaîne.

    Returns:
        Valeur Python typée.
    """
    raw = raw.strip()

    # NULL MySQL → None Python
    if raw.upper() == "NULL":
        return None

    # Valeur entre guillemets simples
    if raw.startswith("'") and raw.endswith("'"):
        inner = raw[1:-1]
        # Dé-échapper les guillemets simples MySQL (\' → ')
        inner = inner.replace("\\'", "'").replace("\\\\", "\\")

        # Tenter un parse JSON si ça ressemble à un objet ou tableau
        stripped = inner.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                logger.debug(f"Impossible de parser comme JSON : {stripped[:80]}...")
        return inner

    # Entier ou float
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        pass

    # Retourne brut si aucune conversion possible
    logger.debug(f"Valeur non typée retournée telle quelle : {raw[:80]}")
    return raw


def _parse_all_rows(sql_text: str, columns: list[str]) -> list[dict]:
    """
    Extrait et parse toutes les lignes VALUES de tous les blocs INSERT du SQL.

    Args:
        sql_text : Contenu brut du fichier SQL.
        columns  : Liste des noms de colonnes.

    Returns:
        Liste de dictionnaires {colonne: valeur}.
    """
    rows = []
    insert_blocks = RE_INSERT_BLOCK.findall(sql_text)

    if not insert_blocks:
        logger.warning("⚠️  Aucun bloc INSERT trouvé dans le SQL.")
        return rows

    logger.info(f"🔍 {len(insert_blocks)} bloc(s) INSERT trouvé(s), parsing en cours...")

    for block_idx, block in enumerate(insert_blocks):
        # Chaque bloc VALUES peut contenir plusieurs lignes séparées par "),\n("
        # On découpe en lignes individuelles
        row_strings = _split_values_into_rows(block)

        for row_str in row_strings:
            # Retire les caractères parasites en tête : ",\n(" entre deux lignes VALUES
            # et les parenthèses extérieures
            row_str = row_str.strip()
            # Cas : ",\n('valeur'..." — trouver la vraie '(' de début de ligne
            if not row_str.startswith("("):
                paren_idx = row_str.find("(")
                if paren_idx != -1:
                    row_str = row_str[paren_idx:]
            if row_str.startswith("("):
                row_str = row_str[1:]
            if row_str.endswith(")"):
                row_str = row_str[:-1]

            raw_values = _split_row_values(row_str)

            if len(raw_values) != len(columns):
                logger.warning(
                    f"Bloc {block_idx} — ligne ignorée : "
                    f"{len(raw_values)} valeurs pour {len(columns)} colonnes. "
                    f"Aperçu : {row_str[:100]}"
                )
                continue

            parsed_values = [_parse_value(v) for v in raw_values]
            rows.append(dict(zip(columns, parsed_values)))

    logger.info(f"✅ {len(rows)} lignes parsées avec succès.")
    return rows


def _split_values_into_rows(values_block: str) -> list[str]:
    """
    Découpe un bloc VALUES en lignes individuelles en tenant compte
    des parenthèses imbriquées (JSON).

    Args:
        values_block : Chaîne contenant toutes les lignes VALUES d'un INSERT.

    Returns:
        Liste de chaînes, chacune représentant une ligne (avec ses parenthèses).
    """
    rows = []
    depth = 0
    current = []
    in_string = False
    escape_next = False

    for char in values_block:
        if escape_next:
            current.append(char)
            escape_next = False
            continue

        if char == "\\" and in_string:
            current.append(char)
            escape_next = True
            continue

        if char == "'" and depth >= 1:
            in_string = not in_string

        if not in_string:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                current.append(char)
                if depth == 0:
                    rows.append("".join(current).strip())
                    current = []
                continue

        current.append(char)

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# FONCTION PRINCIPALE
# ─────────────────────────────────────────────────────────────────────────────

def load_expenses_from_sql(
    sql_path: Optional[Path] = None,
    validate_columns: bool = True,
) -> pd.DataFrame:
    """
    Point d'entrée principal : charge le fichier SQL et retourne un DataFrame brut.

    Args:
        sql_path         : Chemin vers le fichier SQL. Utilise DEFAULT_SQL_FILE si None.
        validate_columns : Si True, vérifie que les colonnes correspondent à EXPECTED_COLUMNS.

    Returns:
        pd.DataFrame avec toutes les lignes de la table `expenses`,
        types basiques castés (dates en datetime, amount en float).

    Raises:
        FileNotFoundError : Si le fichier SQL n'existe pas.
        ValueError        : Si la structure ne correspond pas aux attentes.
    """
    path = Path(sql_path) if sql_path else DEFAULT_SQL_FILE

    logger.info(f"📂 Chargement du fichier SQL : {path}")

    if not path.exists():
        raise FileNotFoundError(f"Fichier SQL introuvable : {path}")

    # Lecture du fichier (encodage utf-8 avec fallback latin-1 pour MySQL)
    try:
        sql_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 échoué, tentative avec latin-1...")
        sql_text = path.read_text(encoding="latin-1")

    logger.info(f"📄 Fichier lu : {len(sql_text):,} caractères")

    # --- Extraction des colonnes ---
    columns = _extract_column_names(sql_text)

    # --- Validation optionnelle ---
    if validate_columns:
        missing = set(EXPECTED_COLUMNS) - set(columns)
        extra   = set(columns) - set(EXPECTED_COLUMNS)
        if missing:
            logger.warning(f"⚠️  Colonnes attendues manquantes : {missing}")
        if extra:
            logger.warning(f"⚠️  Colonnes inattendues présentes : {extra}")

    # --- Parsing des lignes ---
    rows = _parse_all_rows(sql_text, columns)

    if not rows:
        logger.error("❌ Aucune ligne extraite. Vérifiez le format du fichier SQL.")
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows, columns=columns)
    logger.info(f"📊 DataFrame créé : {df.shape[0]} lignes × {df.shape[1]} colonnes")

    # --- Cast des colonnes de dates ---
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            null_count = df[col].isna().sum()
            if null_count > 0:
                logger.debug(f"  → {col} : {null_count} valeurs nulles après conversion")

    # --- Cast du montant ---
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # --- Cast de state ---
    if "state" in df.columns:
        df["state"] = pd.to_numeric(df["state"], errors="coerce").astype("Int64")

    logger.info("✅ Chargement terminé avec succès.")
    logger.info(f"\n{df.dtypes.to_string()}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE RAPIDE POUR TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_expenses_from_sql()
    print(df.head())
    print(df.info())