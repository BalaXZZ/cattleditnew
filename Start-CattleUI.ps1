$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath "$PSScriptRoot\.detector-venv\Scripts\python.exe")) {
    throw 'Create .detector-venv and install dependencies first. See README.md.'
}
if (-not (Test-Path -LiteralPath "$PSScriptRoot\outputs\embeddings\active_model.json")) {
    throw 'Install the team runtime bundle first. See README.md.'
}
& "$PSScriptRoot\.detector-venv\Scripts\python.exe" "$PSScriptRoot\scripts\serve_identity_ui.py" --port 8766
