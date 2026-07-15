# ORBIT-8 v2.1.0 Preview

First downloadable inference release of ORBIT-8 by SeaLandX.

## Included

- Calibrated 16M rhythm Transformer.
- Dynamic Trans-1 arrangement Transformer.
- Automatic BPM and offset ranker.
- Strict two-hand HandFlow optimization.
- FiNALE-compatible `maidata.txt` export.
- Local browser interface with difficulty and pattern controls.

## Requirements

- Windows 10 or 11.
- Python 3.12.
- NVIDIA GPU with a working CUDA 12.8 driver.
- Approximately 6 GB of free disk space for the environment and package.

## Install

1. Download `ORBIT-8-v2.1.0-preview-windows.zip` from this release.
2. Extract the archive.
3. Run `setup_orbit8.ps1` in PowerShell.
4. Run `start_maimai_web.ps1`.
5. Open <http://127.0.0.1:8765/>.

## Scope

This is a research preview. Generated charts must be checked and play-tested
before publication. Training charts, training audio, and intermediate
checkpoints are intentionally excluded. The release contains only the runtime
metadata and inference weights required by ORBIT-8 v2.1.
