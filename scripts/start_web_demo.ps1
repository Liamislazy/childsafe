param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Create the Python 3.12 .venv and install runtime dependencies first.' }
$logRoot = Join-Path $projectRoot 'outputs\web_demo'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$pidPath = Join-Path $logRoot 'server.pid'
if (Test-Path -LiteralPath $pidPath) {
    $previousProcessId = [int](Get-Content -LiteralPath $pidPath)
    $previousProcess = Get-Process -Id $previousProcessId -ErrorAction SilentlyContinue
    if ($previousProcess -and $previousProcess.Path -eq $pythonPath) { throw 'The demo is already running; stop it before starting another server.' }
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) { throw "Port $Port is already in use." }
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot 'Ultralytics'
$process = Start-Process -FilePath $pythonPath -ArgumentList '-u','scripts/web_demo.py','--port',"$Port" -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logRoot 'stdout.log') -RedirectStandardError (Join-Path $logRoot 'stderr.log') -PassThru
$process.Id | Set-Content -LiteralPath $pidPath
Write-Output "Starting ChildSafe at http://127.0.0.1:$Port (process $($process.Id)); startup logs: $logRoot"
