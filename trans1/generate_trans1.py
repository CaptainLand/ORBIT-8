from __future__ import annotations

import sys
from pathlib import Path

import generate_maimai as pipeline

from trans1.model import Trans1Arranger


ROOT = Path(r"D:\trans")
pipeline.ENGINE_NAME = "ORBIT-8 Trans-1"
pipeline.ENGINE_VERSION = "hybrid-v1"
pipeline.ARRANGER_CHECKPOINT = ROOT / "trans1" / "runs" / "trans1_hybrid_v1" / "best.pt"
pipeline.OfficialPatternArranger = Trans1Arranger


def main() -> None:
    if "--designer" not in sys.argv:
        sys.argv.extend(["--designer", "SeaLandX feat. ORBIT-8 Trans-1"])
    pipeline.main()


if __name__ == "__main__":
    main()
