# build.ps1 - Build both standalone executables for the web testing frontend:
#   1. transformer.exe  (the C engine, via zig cc)   -> repo root
#   2. webchat.exe      (the Go server, self-contained via go:embed) -> tools/webchat/
#
# Run from anywhere:  powershell -File tools\webchat\build.ps1
# Then:               .\tools\webchat\webchat.exe          (auto-detects root)
#                     open http://localhost:8080
$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $PSCommandPath          # tools/webchat
$root = Split-Path -Parent (Split-Path -Parent $here)  # repo root

# Pick a C toolchain: $env:ZIG, the bundled zig, or zig/cc on PATH.
$zig = if ($env:ZIG) { $env:ZIG }
       elseif (Test-Path "D:\env\zig\zig.exe") { "D:\env\zig\zig.exe" }
       else { "zig" }

Write-Host "[1/2] transformer.exe  (C engine, $zig cc)"
Push-Location $root
try {
  $cfiles = Get-ChildItem *.c | ForEach-Object { $_.Name }
  & $zig cc -std=c11 -O2 -Wall -o transformer.exe @cfiles -lm
  if ($LASTEXITCODE -ne 0) { throw "transformer build failed ($LASTEXITCODE)" }
} finally { Pop-Location }

Write-Host "[2/2] webchat.exe      (Go server, embedded UI)"
Push-Location $here
try {
  & go build -o webchat.exe .
  if ($LASTEXITCODE -ne 0) { throw "webchat build failed ($LASTEXITCODE)" }
} finally { Pop-Location }

Write-Host ""
Write-Host "Built (both are git-ignored via *.exe):"
Write-Host "  $root\transformer.exe"
Write-Host "  $here\webchat.exe"
Write-Host ""
Write-Host "Run:   .\tools\webchat\webchat.exe          # auto-detects repo root + engine"
Write-Host "Open:  http://localhost:8080"
