# build_gpu.ps1 - Build the CPU and GPU flavours from Windows, where there is no
# make. (Precedent: tools/webchat/build.ps1.)
#
#   powershell -File tools\build_gpu.ps1              # all three artifacts
#   powershell -File tools\build_gpu.ps1 -Only colab  # just the Colab binary
#
# The GPU builds need NO OpenCL SDK, header or import library: gpu.c opens the
# driver with dlopen/LoadLibrary at runtime and resolves entry points by name.
# That is what makes it possible to cross-compile a working GPU binary for Linux
# from a Windows box that has no CUDA and no discrete GPU.

param(
  # all | cpu | gpu | colab
  [string]$Only = "all",
  [string]$Zig = "D:\env\zig\zig.exe"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path $Zig)) {
  $cmd = Get-Command zig -ErrorAction SilentlyContinue
  if ($cmd) { $Zig = $cmd.Source } else { throw "zig not found (looked for $Zig and on PATH)" }
}

$srcs = Get-ChildItem *.c | ForEach-Object { $_.Name }
Write-Host "zig:     $Zig"
Write-Host "sources: $($srcs.Count) .c files"

function Build($label, $outFile, $extraArgs) {
  Write-Host "`n[$label] -> $outFile"
  $args = @("cc") + $extraArgs + @("-std=c11", "-O2", "-Wall", "-o", $outFile) + $srcs + @("-lm")
  & $Zig @args
  if ($LASTEXITCODE -ne 0) { throw "$label build FAILED" }
  $f = Get-Item $outFile
  Write-Host ("  ok  {0:N2} MB" -f ($f.Length / 1MB))
}

if ($Only -eq "all" -or $Only -eq "cpu") {
  Build "CPU (native)" "transformer.exe" @()
}

if ($Only -eq "all" -or $Only -eq "gpu") {
  Build "GPU (native, OpenCL)" "transformer_gpu.exe" @("-DTF_GPU")
  Write-Host "  check it with:  .\transformer_gpu.exe gpu-info"
  Write-Host "                  .\transformer_gpu.exe gpu-bench"
}

if ($Only -eq "all" -or $Only -eq "colab") {
  if (-not (Test-Path build)) { New-Item -ItemType Directory build | Out-Null }
  # glibc pinned low so the binary runs on Colab's Ubuntu 22.04 and anything newer.
  Build "GPU (Linux x86_64, for Colab)" "build/transformer-linux-x86_64-gpu" `
        @("-target", "x86_64-linux-gnu.2.31", "-DTF_GPU")
  Write-Host "  upload build/transformer-linux-x86_64-gpu to Colab"
  Write-Host "  then follow tools/colab_gpu_test.ipynb"
}

Write-Host "`ndone."
