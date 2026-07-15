from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from maimai_ai.audio_analysis import analyze_audio


ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "maimai_web"
OUTPUT_ROOT = ROOT / "output"
UPLOAD_ROOT = ROOT / ".generator_uploads"
MODEL_COMMANDS = {
    "v2.2-handflow": ("-m", "v22.generate_v22"),
    "v2.1-handflow": ("-m", "v2.generate_16m_handflow_dynamic_arranger"),
}
DENSITY_PROFILE_PATH = ROOT / "runtime_data" / "test_density_profile.json"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
INVALID_WINDOWS_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
RESERVED_WINDOWS_NAMES = {
    "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))
}


OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
app = FastAPI(
    title="ORBIT-8",
    description="Neural maimai chart engine by SeaLandX",
    version="2.1.0-preview",
)
generation_lock = asyncio.Lock()
DENSITY_PROFILE = json.loads(DENSITY_PROFILE_PATH.read_text(encoding="utf-8"))


def density_for_level(level: float) -> dict:
    rows = DENSITY_PROFILE["profile"]
    nearest = min(rows, key=lambda row: abs(float(row["level"]) - level))
    return nearest


def clean_song_name(value: str) -> str:
    name = INVALID_WINDOWS_NAME.sub("_", value).strip().rstrip(". ")
    name = re.sub(r"\s+", " ", name)[:100]
    if not name or name.upper() in RESERVED_WINDOWS_NAMES:
        name = "untitled"
    return name


def available_output(name: str) -> Path:
    base = OUTPUT_ROOT / clean_song_name(name)
    candidate = base
    number = 2
    while candidate.exists():
        candidate = OUTPUT_ROOT / f"{base.name}_{number}"
        number += 1
    return candidate


def public_result(folder: Path, report: dict | None = None) -> dict:
    if report is None:
        report_path = folder / "generation_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    relative = folder.name
    return {
        "folder_name": folder.name,
        "folder_path": str(folder),
        "created_at": folder.stat().st_mtime,
        "report": report,
        "files": {
            "maidata": f"/outputs/{relative}/maidata.txt",
            "audio": f"/outputs/{relative}/track.mp3",
            "report": f"/outputs/{relative}/generation_report.json",
        },
    }


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/health")
async def health() -> dict:
    return {
        "status": "ok",
        "engine": "ORBIT-8 by SeaLandX",
        "busy": generation_lock.locked(),
        "output_root": str(OUTPUT_ROOT),
        "python": sys.executable,
        "models": list(MODEL_COMMANDS),
    }


@app.get("/api/results")
async def results() -> dict:
    folders = sorted((path for path in OUTPUT_ROOT.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True)
    return {"results": [public_result(folder) for folder in folders[:12]]}


@app.get("/api/density-profile")
async def density_profile() -> dict:
    return DENSITY_PROFILE


@app.post("/api/generate")
async def generate(
    audio: UploadFile = File(...),
    model: str = Form("v2.2-handflow"),
    title: str = Form(...),
    artist: str = Form("Unknown"),
    bpm: float | None = Form(None),
    offset: float | None = Form(None),
    auto_timing: bool = Form(True),
    level: float = Form(...),
    steps: int = Form(50),
    guidance: float = Form(1.25),
    seed: int = Form(20260701),
    interaction_heat: float = Form(1.0),
    sweep_heat: float = Form(0.7),
    jack_heat: float = Form(1.0),
) -> dict:
    if model not in MODEL_COMMANDS:
        raise HTTPException(400, f"Unknown model: {model}")
    if Path(audio.filename or "").suffix.lower() != ".mp3":
        raise HTTPException(400, "Only MP3 files are supported")
    if not auto_timing:
        if bpm is None or not 30 <= bpm <= 400:
            raise HTTPException(400, "BPM must be between 30 and 400")
        if offset is None or not -10 <= offset <= 60:
            raise HTTPException(400, "Offset must be between -10 and 60 seconds")
    if not 12 <= level <= 15:
        raise HTTPException(400, "Difficulty must be between 12 and 15")
    if not 10 <= steps <= 100:
        raise HTTPException(400, "Sampling steps must be between 10 and 100")
    if not 0.5 <= guidance <= 3:
        raise HTTPException(400, "Guidance must be between 0.5 and 3")
    if any(not 0 <= value <= 2 for value in (interaction_heat, sweep_heat, jack_heat)):
        raise HTTPException(400, "Pattern heat must be between 0 and 2")

    upload_path = UPLOAD_ROOT / f"{uuid.uuid4().hex}.mp3"
    size = 0
    try:
        with upload_path.open("wb") as handle:
            while chunk := await audio.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "MP3 exceeds the 500 MB limit")
                handle.write(chunk)
    finally:
        await audio.close()

    timing_analysis = None
    if auto_timing:
        try:
            timing_analysis = await asyncio.to_thread(analyze_audio, upload_path)
            bpm = float(timing_analysis["bpm"])
            offset = float(timing_analysis["offset"])
        except Exception as error:
            upload_path.unlink(missing_ok=True)
            raise HTTPException(422, f"Automatic BPM/Offset analysis failed: {error}") from error

    assert bpm is not None and offset is not None
    density_row = density_for_level(level)
    target_rms_nps = float(density_row["rms_notes_per_second"])
    density_per_measure = target_rms_nps * 240.0 / bpm

    output = available_output(title or Path(audio.filename or "song").stem)
    command = [
        sys.executable,
        *MODEL_COMMANDS[model],
        str(upload_path),
        "--bpm", str(bpm),
        "--offset", str(offset),
        "--level", str(level),
        "--density", str(density_per_measure),
        "--title", title,
        "--artist", artist or "Unknown",
        "--output", str(output),
        "--steps", str(steps),
        "--guidance", str(guidance),
        "--seed", str(seed),
        "--interaction-heat", str(interaction_heat),
        "--sweep-heat", str(sweep_heat),
        "--jack-heat", str(jack_heat),
    ]
    started = time.monotonic()
    try:
        async with generation_lock:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(ROOT),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20 * 60)
        if process.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip() or stdout.decode("utf-8", errors="replace").strip()
            raise HTTPException(500, message[-2000:] or "Generation failed")
        report_path = output / "generation_report.json"
        if not report_path.exists() or not (output / "maidata.txt").exists() or not (output / "track.mp3").exists():
            raise HTTPException(500, "Generator finished without the required output files")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["source_filename"] = audio.filename
        report["audio"] = str(output / "track.mp3")
        report["web_elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["uploaded_bytes"] = size
        report["automatic_timing"] = auto_timing
        report["web_model"] = model
        report["density_calibration"] = {
            "source": "all-version 12-15 prepared chart RMS profile",
            "target_rms_notes_per_second": target_rms_nps,
            "density_per_measure_at_detected_bpm": density_per_measure,
            "profile_level": density_row["level"],
            "exact_charts": density_row["exact_charts"],
            "nearby_charts": density_row["nearby_charts"],
        }
        if timing_analysis is not None:
            report["timing_analysis"] = timing_analysis
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return public_result(output, report)
    except asyncio.TimeoutError as error:
        raise HTTPException(504, "Generation exceeded 20 minutes") from error
    finally:
        upload_path.unlink(missing_ok=True)
        if output.exists() and not (output / "maidata.txt").exists():
            shutil.rmtree(output, ignore_errors=True)


app.mount("/assets", StaticFiles(directory=WEB_ROOT / "assets"), name="assets")
app.mount("/outputs", StaticFiles(directory=OUTPUT_ROOT), name="outputs")
