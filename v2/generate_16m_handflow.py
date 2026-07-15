from __future__ import annotations

import sys
from pathlib import Path

import generate_maimai as pipeline

from trans1.model import Trans1Arranger
from v2.handflow import optimize_handflow
from v2.rhythm_model_16m import OrbitV2RhythmModel16M


ROOT = Path(r"D:\trans")
pipeline.ENGINE_NAME = "ORBIT-8"
pipeline.ENGINE_VERSION = "v2 16M HandFlow"
pipeline.RHYTHM_CHECKPOINT = ROOT / "v2" / "releases" / "orbit_v2_16m_calibrated.pt"
pipeline.ARRANGER_CHECKPOINT = ROOT / "trans1" / "runs" / "trans1_hybrid_v1" / "best.pt"
pipeline.RhythmPlanModel = OrbitV2RhythmModel16M
pipeline.OfficialPatternArranger = Trans1Arranger
pipeline.HAND_FLOW_OPTIMIZER = optimize_handflow


def main() -> None:
    if "--designer" not in sys.argv:
        sys.argv.extend(["--designer", "SeaLandX feat. ORBIT-8 v2 16M HandFlow"])
    pipeline.main()


if __name__ == "__main__":
    main()
