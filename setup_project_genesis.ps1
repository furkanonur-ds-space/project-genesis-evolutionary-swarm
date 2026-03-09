$ErrorActionPreference = "Stop"

$projectPath = Join-Path -Path (Get-Location) -ChildPath "project_genesis"

New-Item -ItemType Directory -Path $projectPath -Force | Out-Null
Set-Location -Path $projectPath

python -m venv .venv

@"
pygame
numpy
"@ | Set-Content -Path "requirements.txt" -Encoding Ascii

$files = @(
    "main.py",
    "agent.py",
    "brain.py",
    "food.py",
    "genetic_algorithm.py"
)

foreach ($file in $files) {
    New-Item -ItemType File -Path $file -Force | Out-Null
}

Write-Host "Project Genesis scaffolded at $projectPath"
Write-Host "Next steps:"
Write-Host "  1) .\.venv\Scripts\Activate.ps1"
Write-Host "  2) pip install -r requirements.txt"
Write-Host "  3) python main.py"
