# Runs the motion environment's Unity PlayMode tests (roadmap MA) in batch mode.
#   .\unity-env\tools\run-unity-env.ps1                  # all tests
#   .\unity-env\tools\run-unity-env.ps1 -Filter Ladder   # tests whose names match
# Results and the log go to outputs/unity-env/; measured reports to outputs/ma/.
[CmdletBinding()]
param(
    [string]$UnityEditor = $env:SORA_UNITY_EDITOR,
    [string]$Filter = ''
)
$ErrorActionPreference = 'Stop'
$lab = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)  # unity-env\tools -> motion-lab
if (-not $UnityEditor) { $UnityEditor = Join-Path $env:ProgramFiles 'Unity/Hub/Editor/6000.6.0f1/Editor/Unity.exe' }
if (-not (Test-Path -LiteralPath $UnityEditor)) { throw 'Unity 6000.6.0f1 is required. Set SORA_UNITY_EDITOR or pass -UnityEditor.' }
$project = Join-Path $lab 'unity-env'
$evidence = Join-Path $lab 'outputs/unity-env'
New-Item -ItemType Directory -Force $evidence, (Join-Path $lab 'outputs/ma') | Out-Null
$results = Join-Path $evidence 'playmode-results.xml'
$log = Join-Path $evidence 'playmode.log'
if (Test-Path -LiteralPath $results) { Remove-Item -LiteralPath $results }
$arguments = @('-batchmode', '-projectPath', $project, '-logFile', $log, '-runTests', '-testPlatform', 'PlayMode',
    '-testResults', $results, '-assemblyNames', 'MotionEnv.Tests')
if ($Filter) { $arguments += @('-testFilter', $Filter) }
$quoted = $arguments | ForEach-Object { '"' + $_ + '"' }
$process = Start-Process -FilePath $UnityEditor -ArgumentList $quoted -WindowStyle Hidden -PassThru
if (-not $process.WaitForExit(3600000)) { $process.Kill(); throw "Unity exceeded an hour. See $log" }
if ((Get-Content -Raw -LiteralPath $log) -match '(?m)error CS\d+|Scripts have compiler errors') {
    Select-String -LiteralPath $log -Pattern 'error CS\d+|Assembly has|compiler errors' | Select-Object -First 20 | ForEach-Object { $_.Line }
    throw "Compilation failed. See $log"
}
if (-not (Test-Path -LiteralPath $results)) { throw "Unity produced no test results. See $log" }
[xml]$report = Get-Content -Raw -LiteralPath $results
$run = $report.'test-run'
Write-Host "Unity PlayMode: $($run.passed) passed / $($run.failed) failed / $($run.skipped) skipped of $($run.total)"
foreach ($case in $report.SelectNodes('//test-case[@result="Failed"]')) {
    Write-Host "FAILED $($case.name): $($case.failure.message.'#cdata-section')"
}
Select-String -LiteralPath $log -Pattern '^(MA|A[1-4]) ' | ForEach-Object { $_.Line }
