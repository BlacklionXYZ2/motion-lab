# Copies the avatar layer (roadmap MA) from the training environment into SORA's avatar player,
# where it runs her physics body. motion-lab's copy is the source of truth; never edit SORA's.
# SORA's tests fail if the copies drift.
$ErrorActionPreference = 'Stop'
$lab = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)  # unity-env\tools -> motion-lab
$source = Join-Path $lab 'unity-env/Assets/AvatarLayer'
$target = Join-Path (Split-Path -Parent $lab) 'SORA/clients/unity/Assets/Scripts/AvatarLayer'
New-Item -ItemType Directory -Force $target | Out-Null
foreach ($file in Get-ChildItem -LiteralPath $source -Filter '*.cs') {
    Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $target $file.Name) -Force
    Write-Host "copied $($file.Name)"
}
