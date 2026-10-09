$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pidPath = Join-Path $projectRoot 'outputs\web_demo\server.pid'
if (-not (Test-Path -LiteralPath $pidPath)) { throw 'No demo process recorded.' }
$demoProcessId = [int](Get-Content -LiteralPath $pidPath)
$process = Get-Process -Id $demoProcessId -ErrorAction SilentlyContinue
if (-not $process) { Write-Output 'The recorded demo is already stopped.'; exit }
$expectedPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if ($process.Path -ne $expectedPython) { throw 'PID belongs to another executable; refusing to stop it.' }
& taskkill.exe /PID $demoProcessId /T /F
if ($LASTEXITCODE -ne 0) { throw 'Could not stop the demo process.' }
