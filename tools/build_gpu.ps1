# build_gpu.ps1 - Build the CPU and GPU flavours from Windows, where there is no
# make. (Precedent: tools/webchat/build.ps1.)
#
#   powershell -File tools\build_gpu.ps1                   # the default set (below)
#   powershell -File tools\build_gpu.ps1 -Only colab       # just the OpenCL Colab binary
#   powershell -File tools\build_gpu.ps1 -Only colab-cuda  # just the CUDA Colab binary
#
#   -Only    what                                   output
#   cpu      CPU only (native)                      transformer.exe
#   gpu      + OpenCL (native)                      transformer_gpu.exe
#   cuda     + CUDA + OpenCL (native)               transformer_cuda.exe
#   hip      + HIP + OpenCL (native)                transformer_hip.exe
#   allgpu   + every backend (native)               transformer_allgpu.exe
#   colab    Linux x86_64, OpenCL                   build/transformer-linux-x86_64-gpu
#   colab-cuda Linux x86_64, CUDA + OpenCL          build/transformer-linux-x86_64-cuda
#   all      cpu, gpu, cuda, colab, colab-cuda (the default)
#
# The GPU builds need NO SDK, header or import library of any vendor: every
# backend opens its driver/BLAS library with LoadLibrary/dlopen at runtime and
# resolves entry points by name (backend.h). That is what makes it possible to
# cross-compile a working CUDA binary for Linux from a Windows box that has no
# CUDA toolkit and no NVIDIA GPU. A GPU build without the libraries falls back to
# the CPU with one clear line (`gpu-info` shows every backend's status).

param(
  [ValidateSet("all", "cpu", "gpu", "cuda", "hip", "allgpu", "colab", "colab-cuda")]
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

function Want($name) { return ($Only -eq $name) -or ($Only -eq "all" -and $name -in @("cpu", "gpu", "cuda", "colab", "colab-cuda")) }

if (Want "cpu") {
  Build "CPU (native)" "transformer.exe" @()
}

if (Want "gpu") {
  Build "GPU (native, OpenCL)" "transformer_gpu.exe" @("-DTF_GPU")
  Write-Host "  check it with:  .\transformer_gpu.exe gpu-info"
  Write-Host "                  .\transformer_gpu.exe gpu-bench"
}

if (Want "cuda") {
  Build "GPU (native, CUDA + OpenCL)" "transformer_cuda.exe" @("-DTF_CUDA", "-DTF_GPU")
  Write-Host "  needs nvcuda.dll (NVIDIA driver) + cublas64_*.dll (CUDA toolkit) at RUN time;"
  Write-Host "  without them it says why and uses the CPU:  .\transformer_cuda.exe gpu-info"
}

if (Want "hip") {
  Build "GPU (native, HIP + OpenCL)" "transformer_hip.exe" @("-DTF_HIP", "-DTF_GPU")
}

if (Want "allgpu") {
  Build "GPU (native, every backend)" "transformer_allgpu.exe" @("-DTF_CUDA", "-DTF_HIP", "-DTF_GPU")
}

if ((Want "colab") -or (Want "colab-cuda")) {
  if (-not (Test-Path build)) { New-Item -ItemType Directory build | Out-Null }
}

if (Want "colab") {
  # glibc pinned low so the binary runs on Colab's Ubuntu 22.04 and anything newer.
  Build "GPU (Linux x86_64, OpenCL, for Colab)" "build/transformer-linux-x86_64-gpu" `
        @("-target", "x86_64-linux-gnu.2.31", "-DTF_GPU")
  Write-Host "  upload build/transformer-linux-x86_64-gpu to Colab"
  Write-Host "  then follow tools/colab_gpu_test.ipynb"
}

if (Want "colab-cuda") {
  Build "GPU (Linux x86_64, CUDA + OpenCL, for Colab)" "build/transformer-linux-x86_64-cuda" `
        @("-target", "x86_64-linux-gnu.2.31", "-DTF_CUDA", "-DTF_GPU")
  Write-Host "  upload build/transformer-linux-x86_64-cuda to Colab; README 'GPU backends' has the recipe"
}

Write-Host "`ndone."
