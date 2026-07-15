$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& ".\.venv-maimai-ai\Scripts\python.exe" -m uvicorn maimai_web.server:app --host 127.0.0.1 --port 8765
