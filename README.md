# ORBIT-8

[English](README.md) | [简体中文](README.zh-CN.md)

> [!IMPORTANT]
> **Repository source archives do not include model checkpoints.** Download
> `ORBIT-8-v2.2-models.zip` from
> [GitHub Releases](https://github.com/CaptainLand/ORBIT-8/releases), then place
> its two `.pt` files in `v22/releases/`. Training data is intentionally not
> distributed.

**Neural maimai chart engine by SeaLandX.**

ORBIT-8 converts an MP3 and a target difficulty into a FiNALE-compatible chart
folder containing `maidata.txt` and `track.mp3`. It estimates BPM and offset,
extracts an audio-aligned rhythm plan, arranges playable maimai patterns, and
checks the result against two-hand movement constraints before export.

> ORBIT-8 is a research prototype. Generated charts should be reviewed and
> play-tested in an editor before publication.

## Architecture

```mermaid
flowchart LR
    A["MP3 + target difficulty"] --> B["Audio analysis<br/>BPM, offset, onset features"]
    B --> C["16M rhythm Transformer<br/>timing and note density"]
    C --> D["Dynamic arrangement Transformer<br/>lanes, interactions, sweeps and slides"]
    D --> E["Pattern and slide planner<br/>official-chart-derived vocabulary"]
    E --> F["HandFlow optimizer<br/>left/right hand tracking"]
    F --> G["Playability validator<br/>capacity, paths, tails, holds and KPS"]
    G --> H["Simai exporter<br/>maidata.txt + track.mp3"]
```

ORBIT-8 deliberately separates **rhythm transcription** from **chart
arrangement**. The rhythm model decides *when* notes should occur by combining
audio features with the requested difficulty. The arrangement model decides
*how* those notes should be expressed on the eight-button ring, learning lane
movement and pattern vocabulary from level 12-15 official charts.

The neural output then passes through a deterministic playability layer:

- **HandFlow beam search** tracks both hands, their current lanes, travel speed,
  crossed posture, and active hold or slide reservations.
- **Pattern-aware constraints** preserve intentional interactions and regular
  sweeps while repairing irregular 16th-note hand changes and excessive jacks.
- **Long-object safety** treats holds, slides, and Wi-Fi slides as occupied hands
  and protects slide paths and tails from taps, collisions, and impossible chords.
- **Difficulty calibration** controls density, interaction, sweep, and jack heat
  before producing a chart that can be inspected in a maimai editor.

## Model Line

| Model | Rhythm planning | Arrangement | Playability |
| --- | --- | --- | --- |
| ORBIT-8 v1.7.1 | consensus onset pipeline | official-pattern arranger | rule validation |
| Trans-02 | rhythm Transformer | Trans-1 Transformer | rule validation |
| ORBIT-8 v2.1 HandFlow | calibrated 16M Transformer | dynamic Trans-1 Transformer | strict two-hand HandFlow |
| ORBIT-8 v2.2 HandFlow | held-out calibrated 16M Transformer | scheduled-sampling Trans arranger | full-song HandFlow acceptance |

The current experimental mainline is **ORBIT-8 v2.2 HandFlow**. It retains the
16M rhythm backbone, applies held-out onset calibration, and trains its arranger
with scheduled sampling so inference more closely matches training. The released
arranger contains about 3M parameters. Full-song acceptance then checks hand
capacity, long-object reservations, slide paths, tails, and irregular hand flow.

## Pipeline

1. Analyze audio and estimate BPM, offset, and onset features.
2. Generate a difficulty-conditioned rhythm plan aligned to the music.
3. Predict lanes and maimai pattern operators with the arrangement Transformer.
4. Construct taps, holds, interactions, sweeps, and legal slide templates.
5. Optimize left/right hand assignments and repair unreasonable movement.
6. Validate hand capacity, long-object conflicts, slide paths, tails, and density.
7. Export a song folder containing `track.mp3`, `maidata.txt`, and a generation report.

Generated charts use `SeaLandX feat. ORBIT-8` as the default designer credit.

## Download And Run

The v2.2 preview currently requires Windows, Python 3.12, and an NVIDIA GPU with
a working CUDA 12.8 driver.

1. Clone this repository or download its source archive.
2. Download `ORBIT-8-v2.2-models.zip` from
   [Releases](https://github.com/CaptainLand/ORBIT-8/releases), then copy
   `orbit_v22_rhythm.pt` and `orbit_v22_arranger.pt` into `v22/releases/`.
3. Open PowerShell in the project folder and install the environment:

```powershell
.\setup_orbit8.ps1
```

4. Start the local web interface:

```powershell
.\start_maimai_web.ps1
```

Open <http://127.0.0.1:8765/> and import an MP3. The web interface supports model
selection and controls for difficulty, interaction, sweep, and jack intensity.

## Training Data Notice

Training datasets, official chart archives, copyrighted audio, generated songs,
virtual environments, and intermediate checkpoints are not distributed. The
release package contains only the inference weights, evaluation report, and
compact runtime metadata required to run ORBIT-8 v2.2.
