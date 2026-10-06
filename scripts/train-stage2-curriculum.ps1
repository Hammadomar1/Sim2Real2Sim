param([string]$RunDir='runs/stage2_distance_seed73',[string]$Resume='',[int]$Updates=100,[double]$Hours=0.5,[int]$NumEnvs=2048)
$ErrorActionPreference='Stop'
$projectRoot=Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$linuxPath=(& wsl -d Ubuntu-22.04 -- wslpath -a ($projectRoot.Replace('\','/'))).Trim()
$parameters=@('--run-dir',$RunDir,'--updates',"$Updates",'--hours',"$Hours",'--num-envs',"$NumEnvs")
if ($Resume) { $parameters+=@('--resume',$Resume) }
& wsl -d Ubuntu-22.04 -- /home/$env:USERNAME/.venvs/so101-m1/bin/python -u "$linuxPath/scripts/train_distance_curriculum.py" @parameters
if ($LASTEXITCODE -ne 0) { throw 'Distance curriculum stopped with an error; inspect the run directory.' }
