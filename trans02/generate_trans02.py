from __future__ import annotations

import sys
from pathlib import Path

import generate_maimai as pipeline

from trans1.model import Trans1Arranger
from trans02.rhythm_model import Trans02RhythmModel


ROOT = Path(r"D:\trans")
pipeline.ENGINE_NAME = "ORBIT-8 Trans-02"
pipeline.ENGINE_VERSION = "hybrid-v1"
pipeline.RHYTHM_CHECKPOINT = ROOT / "trans02" / "runs" / "trans02_rhythm_hybrid_v1" / "best.pt"
pipeline.ARRANGER_CHECKPOINT = ROOT / "trans1" / "runs" / "trans1_hybrid_v1" / "best.pt"
pipeline.RhythmPlanModel = Trans02RhythmModel
pipeline.OfficialPatternArranger = Trans1Arranger


def main() -> None:
    if "--designer" not in sys.argv:
        sys.argv.extend(["--designer", "SeaLandX feat. ORBIT-8 Trans-02"])
    pipeline.main()


if __name__ == "__main__":
    main()
