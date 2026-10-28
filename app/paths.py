"""Where things are: the repo root and the Qwenputer weights (QWENPUTER_MODEL, default models/Qwen3.5-4B-Qwenputer)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = Path(os.environ.get("QWENPUTER_MODEL", ROOT / "models" / "Qwen3.5-4B-Qwenputer"))
