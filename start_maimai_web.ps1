$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$pythonCandidates = @(
    ".\.venv\Scripts\python.exe",
    ".\.venv-maimai-ai\Scripts\python.exe"
)
$python = $pythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $python) {
    throw "Python environment not found. Run .\setup_orbit8.ps1 first."
}

$requiredFiles = @(
    ".\v2\releases\orbit_v2_16m_calibrated.pt",
    ".\trans1\releases\trans1_dynamic_v2.pt",
    ".\runtime_data\bpm_ranker.joblib"
)
$missing = $requiredFiles | Where-Object { -not (Test-Path $_) }
if ($missing) {
    throw "Release model files are missing: $($missing -join ', '). Download the full GitHub Release package."
}

& $python -m uvicorn maimai_web.server:app --host 127.0.0.1 --port 8765
