param(
    [ValidateSet("both", "idle", "walk")]
    [string]$Mode = "both",
    [int]$IdleSeconds = 60,
    [int]$WalkSeconds = 60,
    [string]$ExePath = "$PSScriptRoot\..\target\release\pi-avatar.exe"
)

$ErrorActionPreference = 'Stop'
$ExePath = (Resolve-Path $ExePath).Path
$LogicalCores = [Environment]::ProcessorCount

function Measure-Run {
    param(
        [string]$Label,
        [string]$ModeFlag,
        [int]$DurationSec
    )
    $statsPath = Join-Path $env:TEMP "rune-pet-stats-$Label.json"
    Remove-Item -Force $statsPath -ErrorAction SilentlyContinue

    $stayMs = ($DurationSec + 6) * 1000
    $proc = Start-Process -FilePath $ExePath -ArgumentList @($ModeFlag, "--stay-open-ms", "$stayMs", "--stats-file", $statsPath) -PassThru
    try {
        # Wait 3s for startup + GPU init + initial pose settle
        Start-Sleep -Seconds 3
        $proc.Refresh()
        if ($proc.HasExited) {
            throw "$Label process exited early with code $($proc.ExitCode)"
        }
        $t0 = [DateTime]::UtcNow
        $cpu0 = $proc.TotalProcessorTime.TotalSeconds

        Start-Sleep -Seconds $DurationSec
        $proc.Refresh()
        if ($proc.HasExited) {
            throw "$Label process exited before measurement completed"
        }

        $t1 = [DateTime]::UtcNow
        $cpu1 = $proc.TotalProcessorTime.TotalSeconds
        $wall = ($t1 - $t0).TotalSeconds
        $cpuDelta = $cpu1 - $cpu0
        $cpuPctTotal = ($cpuDelta / ($wall * $LogicalCores)) * 100.0
        $cpuPctOneCore = ($cpuDelta / $wall) * 100.0

        $gp = Get-Process -Id $proc.Id
        $wsMb = [Math]::Round($gp.WorkingSet64 / 1MB, 2)
        $privMb = [Math]::Round($gp.PrivateMemorySize64 / 1MB, 2)
        $tl = (tasklist /FI "PID eq $($proc.Id)" /FO CSV /NH) | Out-String

        # Wait for graceful exit so stats file is flushed
        if (-not $proc.WaitForExit(10000)) {
            $proc.Kill()
            $proc.WaitForExit(3000) | Out-Null
        }

        $stats = $null
        if (Test-Path $statsPath) {
            $stats = Get-Content $statsPath -Raw | ConvertFrom-Json
            Remove-Item -Force $statsPath -ErrorAction SilentlyContinue
        }

        [PSCustomObject]@{
            Mode            = $Label
            DurationSec     = [Math]::Round($wall, 2)
            WorkingSetMB    = $wsMb
            PrivateMB       = $privMb
            TasklistRow     = $tl.Trim()
            CpuSec          = [Math]::Round($cpuDelta, 3)
            CpuPctAllCores  = [Math]::Round($cpuPctTotal, 3)
            CpuPctOneCore   = [Math]::Round($cpuPctOneCore, 3)
            FramesRendered  = if ($stats) { $stats.frames } else { $null }
            AvgFps          = if ($stats) { $stats.avg_fps } else { $null }
            MinFps          = if ($stats) { $stats.min_fps } else { $null }
            P5Fps           = if ($stats) { $stats.p5_fps } else { $null }
        }
    } finally {
        if (-not $proc.HasExited) {
            try { $proc.Kill() } catch {}
        }
    }
}

$idleRes = if ($Mode -in @("both", "idle")) { Measure-Run -Label "idle" -ModeFlag "--force-idle" -DurationSec $IdleSeconds } else { $null }
$walkRes = if ($Mode -in @("both", "walk")) { Measure-Run -Label "walk" -ModeFlag "--force-walk" -DurationSec $WalkSeconds } else { $null }

$out = [PSCustomObject]@{
    BinaryPath  = $ExePath
    BinaryBytes = (Get-Item $ExePath).Length
    BinaryMB    = [Math]::Round((Get-Item $ExePath).Length / 1MB, 2)
    Cores       = $LogicalCores
    Idle        = $idleRes
    Walk        = $walkRes
}

$out | ConvertTo-Json -Depth 5
