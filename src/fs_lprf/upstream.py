"""Locate the unmodified, pinned Layer-Wise submodule."""
from pathlib import Path
import sys

UPSTREAM_COMMIT = "88efe1376dfc5f5b9b62d016d33a7381c158d8af"
XLSR_URL = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt"
XLSR_BYTES = 3808868242
XLSR_SHA256 = "b08927597f2c9eb2ebd7dcc3ac78ee4b5f6021cbac4b3a6c5a9deec445d80ed9"
ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party" / "tomatoDD"


def enable_upstream():
    if not (UPSTREAM / "tomato" / "models" / "frontend_only.py").is_file():
        raise RuntimeError("Missing Layer-Wise submodule: git submodule update --init --recursive")
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
