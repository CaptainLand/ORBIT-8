$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv .venv
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv .venv
    }
    else {
        throw "Python 3.12 was not found. Install it from https://www.python.org/downloads/"
    }
}

$python = ".\.venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install -r requirements.txt

Write-Host ""
Write-Host "ORBIT-8 setup complete. Run .\start_maimai_web.ps1"
