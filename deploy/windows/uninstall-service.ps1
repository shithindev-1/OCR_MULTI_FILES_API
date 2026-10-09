<#
.SYNOPSIS
    Stops and removes the service. The code, .venv, .env and logs are left in place.

        powershell -ExecutionPolicy Bypass -File deploy\windows\uninstall-service.ps1
#>
[CmdletBinding()]
param(
    [string]$ServiceName = "UaeOcrApi",
    [string]$NssmPath = "nssm"
)

$ErrorActionPreference = "Stop"

& $NssmPath stop $ServiceName confirm
& $NssmPath remove $ServiceName confirm
