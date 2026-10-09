<#
.SYNOPSIS
    Installs (or reinstalls) the UAE OCR API as a Windows service with NSSM.

.DESCRIPTION
    Run from an elevated PowerShell on the VM, after cloning the repo and filling in .env:

        powershell -ExecutionPolicy Bypass -File deploy\windows\install-service.ps1

    Creates .venv if it is missing, installs requirements.txt, then registers a service that
    runs `.venv\Scripts\python.exe -m app` from the project folder, starts at boot, restarts
    itself after a crash, and writes its console output to logs\service-*.log.
#>
[CmdletBinding()]
param(
    [string]$ServiceName = "UaeOcrApi",
    [string]$NssmPath = "nssm",
    [string]$PythonLauncher = "py"
)

$ErrorActionPreference = "Stop"

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this script from an elevated (Administrator) PowerShell."
}

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogDir = Join-Path $ProjectRoot "logs"

if (-not (Get-Command $NssmPath -ErrorAction SilentlyContinue)) {
    throw "NSSM not found. Install it (e.g. 'choco install nssm' or 'winget install NSSM.NSSM') or pass -NssmPath C:\path\to\nssm.exe."
}
if (-not (Test-Path (Join-Path $ProjectRoot ".env"))) {
    throw "No .env in $ProjectRoot. Copy .env.example to .env and fill it in first."
}

if (-not (Test-Path $Python)) {
    Write-Host "Creating virtual environment..."
    & $PythonLauncher -m venv (Join-Path $ProjectRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv" }
}

Write-Host "Installing dependencies..."
& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) {
    Write-Host "Removing existing $ServiceName service..."
    & $NssmPath stop $ServiceName confirm | Out-Null
    & $NssmPath remove $ServiceName confirm | Out-Null
}

Write-Host "Registering $ServiceName..."
& $NssmPath install $ServiceName $Python "-m" "app"
& $NssmPath set $ServiceName AppDirectory $ProjectRoot
& $NssmPath set $ServiceName DisplayName "UAE OCR API"
& $NssmPath set $ServiceName Description "FastAPI OCR service for UAE identity and leasing documents."
& $NssmPath set $ServiceName Start SERVICE_AUTO_START
# Unbuffered output, so the service log files are written in real time.
& $NssmPath set $ServiceName AppEnvironmentExtra "PYTHONUNBUFFERED=1"
# Restart after any exit, waiting 5 s between attempts so a bad config cannot spin the CPU.
& $NssmPath set $ServiceName AppExit Default Restart
& $NssmPath set $ServiceName AppRestartDelay 5000
& $NssmPath set $ServiceName AppThrottle 10000
# Give Uvicorn time to finish in-flight requests on stop.
& $NssmPath set $ServiceName AppStopMethodConsole 15000
& $NssmPath set $ServiceName AppStdout (Join-Path $LogDir "service-stdout.log")
& $NssmPath set $ServiceName AppStderr (Join-Path $LogDir "service-stderr.log")
& $NssmPath set $ServiceName AppRotateFiles 1
& $NssmPath set $ServiceName AppRotateOnline 1
& $NssmPath set $ServiceName AppRotateBytes 10485760

& $NssmPath start $ServiceName
Start-Sleep -Seconds 3
& $NssmPath status $ServiceName
Write-Host "Done. Check: Invoke-RestMethod http://127.0.0.1:<APP_PORT>/health"
