# ジョブ一覧を回し終わるまで、この PC がアイドルでスリープしないようにする（Windows の SetThreadExecutionState）。
# 電源設定は変えない。このスクリプトが動いている間だけ有効で、終わると元に戻る。ふたを閉じる・手動でスリープすると止まる。
# 使い方（リポジトリのルートで）。WMI 経由で起動すると、起動したアプリやターミナルを閉じても止まらない:
#   $cmd = "powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$PWD\results\hh_improve\run_awake.ps1`" results/hh_improve/jobs_cpu_round9.txt 1"
#   Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine = $cmd; CurrentDirectory = "$PWD"}
param(
    [Parameter(Mandatory = $true)][string]$Jobs,
    [int]$Parallel = 1
)
$root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
Set-Location $root
Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);
'@
$ES_CONTINUOUS = [uint32]'0x80000000'
$ES_SYSTEM_REQUIRED = [uint32]'0x00000001'
$name = [IO.Path]::GetFileNameWithoutExtension($Jobs)
New-Item -ItemType Directory -Force results\hh_improve\logs | Out-Null
$out = "results\hh_improve\logs\$name.out"
$bash = 'C:\Program Files\Git\bin\bash.exe'
if (-not (Test-Path $bash)) { $bash = 'bash' }
[Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED) | Out-Null
try {
    # ログは bash に書かせる（PowerShell 5 のリダイレクトは UTF-16 になるため）
    $o = $out -replace '\\', '/'
    & $bash -c "echo start `$(date +%FT%T) jobs=$Jobs parallel=$Parallel > $o; bash results/hh_improve/run_jobs.sh $Jobs $Parallel >> $o 2>&1; echo end `$(date +%FT%T) >> $o"
}
finally {
    [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS) | Out-Null
}
