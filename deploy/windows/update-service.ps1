<#
.SYNOPSIS
    Pulls the latest code, reinstalls dependencies and restarts the service.

        powershell -ExecutionPolicy Bypass -File deploy\windows\update-service.ps1
#>
[CmdletBinding()]
param(
    [string]$ServiceName = "UaeOcrApi",
    [string]$NssmPath = "nssm",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

git -C $ProjectRoot pull --ff-only origin $Branch
if ($LASTEXITCODE -ne 0) { throw "git pull failed" }

& $Python -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

& $NssmPath restart $ServiceName
Start-Sleep -Seconds 3
& $NssmPath status $ServiceName
