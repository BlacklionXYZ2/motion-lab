# Copies SORA's avatar-player code that the motion environment shares (roadmap MA, A1: hair,
# clothing and accessories) into unity-env/Assets/FromSora. SORA's copy is the source of truth;
# never edit the copies here. The environment's tests fail if the copies drift.
$ErrorActionPreference = 'Stop'
$lab = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)  # unity-env\tools -> motion-lab
$sora = Join-Path (Split-Path -Parent $lab) 'SORA/clients/unity/Assets/Scripts/Avatar'
$target = Join-Path $lab 'unity-env/Assets/FromSora'
foreach ($name in 'AvatarSecondaryParts.cs', 'AvatarSecondaryMotionTuner.cs') {
    Copy-Item -LiteralPath (Join-Path $sora $name) -Destination (Join-Path $target $name) -Force
    Write-Host "copied $name"
}
