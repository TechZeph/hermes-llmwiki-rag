<#
.SYNOPSIS
  llmwiki installer for Windows PowerShell: virtualenv + package + first-run setup.
.EXAMPLE
  .\install.ps1
  .\install.ps1 -NoInit          # from a checkout, skip `llmwiki init`
  .\install.ps1 -DryRun
.NOTES
  Requires Python 3.11+ from python.org (its sqlite3 includes FTS5). Hermes on
  Windows runs under WSL, so use install.sh --hermes inside WSL for the plugin.
  Environment overrides: LLMWIKI_INSTALL_DIR, LLMWIKI_PACKAGE, PYTHON.
#>
param([switch]$NoInit, [switch]$DryRun)
$ErrorActionPreference = "Stop"

$InstallDir = if ($env:LLMWIKI_INSTALL_DIR) { $env:LLMWIKI_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA "llmwiki\venv" }
$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
if ($env:LLMWIKI_PACKAGE) { $PackageArgs = @($env:LLMWIKI_PACKAGE); $Source = "explicit" }
elseif ((Test-Path (Join-Path $ScriptDir "pyproject.toml")) -and (Select-String -Quiet -Path (Join-Path $ScriptDir "pyproject.toml") -Pattern 'name = "hermes-llmwiki-rag"')) { $PackageArgs = @("-e", "$ScriptDir[mcp]"); $Source = "checkout ($ScriptDir)" }
else { $PackageArgs = @("hermes-llmwiki-rag[mcp]"); $Source = "PyPI" }

function Say($m) { Write-Host "==> $m" }
function Run { param([string[]]$Cmd) if ($DryRun) { Write-Host "    would run: $($Cmd -join ' ')" } else { & $Cmd[0] $Cmd[1..($Cmd.Length-1)]; if ($LASTEXITCODE -ne 0) { throw "command failed: $($Cmd -join ' ')" } } }

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) { throw "python not found. Install Python 3.11+ from https://www.python.org/downloads/ and re-run." }
$ver = & $Python -c "import sys; print('%d.%d' % sys.version_info[:2])"
& $Python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) { throw "Python $ver found; llmwiki needs 3.11 or newer." }
& $Python -c "import sqlite3; c=sqlite3.connect(':memory:'); c.execute('CREATE VIRTUAL TABLE t USING fts5(x)')" 2>$null
if ($LASTEXITCODE -ne 0) { throw "This Python's sqlite3 lacks FTS5; use the python.org build." }
Say "Python $ver; package source: $Source"
Say "virtualenv: $InstallDir"

if (-not (Test-Path (Join-Path $InstallDir "Scripts\python.exe"))) { Run @($Python, "-m", "venv", $InstallDir) }
$VPy = Join-Path $InstallDir "Scripts\python.exe"
Run @($VPy, "-m", "pip", "install", "--quiet", "--upgrade", "pip")
Run (@($VPy, "-m", "pip", "install", "--quiet", "--upgrade") + $PackageArgs)
$Launcher = Join-Path $InstallDir "Scripts\llmwiki.exe"
Say "launcher: $Launcher (add $(Join-Path $InstallDir 'Scripts') to PATH to run 'llmwiki' directly)"

if ($NoInit) { Say "skipped init; run: $Launcher init" }
else { Say "first-run setup (llmwiki init)"; Run @($Launcher, "init") }
Say "done. Check everything with: $Launcher doctor"
