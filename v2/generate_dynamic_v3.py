from __future__ import annotations

import sys
from pathlib import Path

import generate_maimai as pipeline

from trans1.model import Trans1Arranger
from v2.rhythm_model import OrbitV2RhythmModel


ROOT = Path(r"D:\trans")
pipeline.ENGINE_NAME = "ORBIT-8"
pipeline.ENGINE_VERSION = "v2 dynamic-v3 trial"
pipeline.RHYTHM_CHECKPOINT = ROOT / "v2" / "runs" / "orbit_v2_dynamic_v3_fair_trial" / "best.pt"
pipeline.ARRANGER_CHECKPOINT = ROOT / "trans1" / "runs" / "trans1_hybrid_v1" / "best.pt"
pipeline.RhythmPlanModel = OrbitV2RhythmModel
pipeline.OfficialPatternArranger = Trans1Arranger


def main() -> None:
    if "--designer" not in sys.argv:
        sys.argv.extend(["--designer", "SeaLandX feat. ORBIT-8 v2 dynamic-v3"])
    pipeline.main()


if __name__ == "__main__":
    main()
