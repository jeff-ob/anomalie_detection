"""
app.py — Point d'entrée racine pour le déploiement Streamlit Cloud
Redirige automatiquement vers le dashboard principal situé dans dashboard/app.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import runpy
runpy.run_path(str(ROOT / "dashboard" / "app.py"), run_name="__main__")
