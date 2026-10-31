param([ValidateRange(1024, 65535)][int]$Port = 8002)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Create the Python .venv and install runtime dependencies first.'
}
$url = "http://127.0.0.1:$Port"
function Test-PlayAreaServer {
    try {
        $health = Invoke-RestMethod -Uri "$url/api/play-area/health" -TimeoutSec 2
        return $health.service -eq 'childsafe-play-area' -and $health.status -eq 'running'
    } catch { return $false }
}
if (Test-PlayAreaServer) {
    Write-Output "ChildSafe is already running: $url/play-area"
    exit 0
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is occupied by another server. No process was stopped."
}
$logRoot = Join-Path $projectRoot 'outputs\play-area-server'
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot 'Ultralytics'
$process = Start-Process -FilePath $pythonPath -ArgumentList @('-u', '-X', 'utf8', 'scripts/play_area_demo.py', '--port', "$Port") -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logRoot 'stdout.log') -RedirectStandardError (Join-Path $logRoot 'stderr.log') -PassThru
$process.Id | Set-Content -LiteralPath (Join-Path $logRoot 'server.pid')
$deadline = (Get-Date).AddSeconds(30)
do {
    if (Test-PlayAreaServer) {
        Write-Output "ChildSafe is running in the background: $url/play-area"
        Write-Output "Logs: $logRoot"
        exit 0
    }
    if ($process.HasExited) { throw "ChildSafe stopped during startup. Read $logRoot\stderr.log" }
    Start-Sleep -Milliseconds 300
} while ((Get-Date) -lt $deadline)
throw "ChildSafe has not responded within 30 seconds. Read $logRoot\stderr.log before retrying."
