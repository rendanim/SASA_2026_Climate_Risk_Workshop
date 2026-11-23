import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYNTH = ROOT / "data_aifs_synthetic"


def pytest_configure(config):
    """Tests run on the synthetic bundle unless AIFS_DATA points elsewhere (build it if missing)."""
    if "AIFS_DATA" not in os.environ:
        if not (SYNTH / "meta.json").exists():
            subprocess.run([sys.executable, str(ROOT / "aifs_app" / "prepare_data.py"), "--synthetic"], check=True)
        os.environ["AIFS_DATA"] = str(SYNTH)
