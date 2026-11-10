"""Chemins du projet, résolus depuis la racine du dépôt (indépendants du dossier courant)."""

from pathlib import Path

# racine du dépôt : src/attrition_esn/config.py -> remonte de 2 niveaux
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# crée les dossiers de sortie s'ils n'existent pas (ex : après un clone)
for _dir in (PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
