from __future__ import annotations

import sys
from pathlib import Path

import generate_maimai as pipeline

from trans1.model import Trans1Arranger
from v2.handflow import optimize_handflow
from v2.rhythm_model_16m import OrbitV2RhythmModel16M


ROOT = Path(__file__).resolve().parents[1]
pipeline.ENGINE_NAME = "ORBIT-8"
pipeline.ENGINE_VERSION = "v2.2 Calibrated HandFlow"
pipeline.RHYTHM_CHECKPOINT = ROOT / "v22" / "releases" / "orbit_v22_rhythm.pt"
pipeline.ARRANGER_CHECKPOINT = ROOT / "v22" / "releases" / "orbit_v22_arranger.pt"
pipeline.RhythmPlanModel = OrbitV2RhythmModel16M
pipeline.OfficialPatternArranger = Trans1Arranger
pipeline.HAND_FLOW_OPTIMIZER = optimize_handflow


def main() -> None:
    if "--designer" not in sys.argv:
        sys.argv.extend(["--designer", "SeaLandX feat. ORBIT-8 v2.2 HandFlow"])
    pipeline.main()


if __name__ == "__main__":
    main()
