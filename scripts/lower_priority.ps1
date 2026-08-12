# 降低 python 实验进程优先级, 避免电脑卡顿
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    $_.PriorityClass = 'BelowNormal'
    Write-Host ("PID " + $_.Id + " -> BelowNormal")
}
Write-Host "done"
