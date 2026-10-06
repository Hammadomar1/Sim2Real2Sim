param([ValidateSet('0.002','0.001')][string]$Timestep = '0.001',[switch]$VerifyOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$linuxRoot = (& wsl -d Ubuntu-22.04 -- wslpath -a $projectRoot).Trim()
$suffix = if ($Timestep -eq '0.002') { '2ms' } else { '1ms' }
if (-not $VerifyOnly) {
    & wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python "$linuxRoot/scripts/feedback_path.py" --timestep $Timestep
    if ($LASTEXITCODE -ne 0) { throw 'Feedback planner failed.' }
}
& wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python "$linuxRoot/scripts/validate_path.py" --timestep $Timestep --input "gripper_feedback_$suffix" --output "gripper_feedback_validation_$suffix"
if ($LASTEXITCODE -ne 0) { throw 'Independent feedback-path verification failed.' }
