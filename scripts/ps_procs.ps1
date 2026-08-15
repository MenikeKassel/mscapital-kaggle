Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
    $cpu = [math]::Round(($_.KernelModeTime + $_.UserModeTime) / 1e7, 1)
    $cmd = $_.CommandLine
    if ($cmd -and $cmd.Length -gt 90) { $cmd = $cmd.Substring(0, 90) }
    Write-Output ("{0} | {1}s | {2}" -f $_.ProcessId, $cpu, $cmd)
}
