<#
    Install board_daemon.py as a Windows scheduled task so the tunnel is restored
    automatically at logon and after every replug, with no console window.

    Run from the repo root in PowerShell:

        .\scripts\install_autoconnect.ps1
        .\scripts\install_autoconnect.ps1 -Uninstall

    Requires: Python 3 on PATH, and adb (Android platform-tools) on PATH.
    The API key is read from %USERPROFILE%\.robodog\env, never from this repo.
#>
param([switch]$Uninstall)

$ErrorActionPreference = 'Stop'
$TaskName  = 'RoboDogAutoconnect'
$Root      = Split-Path -Parent $PSScriptRoot
$ConfigDir = Join-Path $env:USERPROFILE '.robodog'
$Runner    = Join-Path $ConfigDir 'board_daemon.py'

if ($Uninstall) {
    schtasks /Delete /TN $TaskName /F 2>$null | Out-Null
    Write-Host "uninstalled $TaskName"
    exit 0
}

$envFile = Join-Path $ConfigDir 'env'
if (-not (Test-Path $envFile)) {
    Write-Error "Create $envFile first, containing:`n  CEREBRAS_API_KEY=csk-..."
    exit 1
}

# pythonw runs without opening a console window; fall back to python if absent.
$py = (Get-Command pythonw.exe -ErrorAction SilentlyContinue)
if (-not $py) { $py = Get-Command python.exe -ErrorAction SilentlyContinue }
if (-not $py) { Write-Error 'Python not found on PATH.'; exit 1 }

if (-not (Get-Command adb.exe -ErrorAction SilentlyContinue)) {
    Write-Warning 'adb.exe is not on PATH. Install Android platform-tools, or the daemon will idle.'
}

New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null
Copy-Item (Join-Path $Root 'scripts\board_daemon.py') $Runner -Force

# /RL LIMITED keeps it in the user session, which is where adb's device access lives.
schtasks /Create /TN $TaskName /SC ONLOGON /RL LIMITED /F `
    /TR "`"$($py.Source)`" `"$Runner`"" | Out-Null
schtasks /Run /TN $TaskName | Out-Null

Write-Host "installed $TaskName"
Write-Host "  log:       Get-Content -Wait $ConfigDir\autoconnect.log"
Write-Host "  uninstall: .\scripts\install_autoconnect.ps1 -Uninstall"
