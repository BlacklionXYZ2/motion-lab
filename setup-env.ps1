[CmdletBinding()]
param(
    # The Python that already has PyTorch with ROCm installed (the owner's Store Python 3.12).
    [string]$Python = "$env:LOCALAPPDATA\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe"
)

# Creates motion-lab's .venv once. It sees the owner's existing PyTorch (ROCm), numpy,
# onnxruntime and pytest through --system-site-packages, so nothing large is downloaded again,
# and adds only onnx, onnxscript and ruff. Windows PowerShell 5.1 compatible.

$ErrorActionPreference = "Stop"
$lab = $PSScriptRoot
$venv = Join-Path $lab ".venv"
if (-not (Test-Path -LiteralPath (Join-Path $venv "Scripts\python.exe"))) {
    & $Python -m venv --system-site-packages $venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create $venv" }
}
$py = Join-Path $venv "Scripts\python.exe"
& $py -m pip install --quiet --disable-pip-version-check onnx onnxscript ruff
if ($LASTEXITCODE -ne 0) { throw "Could not install onnx, onnxscript and ruff" }
# No path file for motionlab: each version folder (versions\vN) has its own copy, and its
# scripts and tests import that copy, so versions never mix.
& $py -c "import torch; print('PyTorch', torch.__version__, '- GPU visible:', torch.cuda.is_available())"
