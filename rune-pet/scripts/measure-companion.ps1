<#
Measures the pet while a scripted 20 s conversation is replayed (`--companion-replay`), against
an idle baseline. Samples the working set and CPU every second, so a RAM peak is not hidden by
the final reading. After the stream ends the run goes on for -TailSeconds of plain idle.

  pwsh -NoProfile -File scripts\measure-companion.ps1 [-TailSeconds 12] [-ExePath ...]
#>
param(
    [int]$TailSeconds = 12,
    [string]$ExePath = "$PSScriptRoot\..\target\release\pi-avatar.exe",
    [string]$Replay = "$PSScriptRoot\talk-20s.jsonl"
)

$ErrorActionPreference = 'Stop'
$ExePath = (Resolve-Path $ExePath).Path
$Replay = (Resolve-Path $Replay).Path
$cores = [Environment]::ProcessorCount

function Measure-Run {
    param([string]$Label, [string[]]$ExtraArgs, [int]$MeasureSec)
    $stats = Join-Path $env:TEMP "rune-pet-companion-$Label.json"
    Remove-Item -Force $stats -ErrorAction SilentlyContinue
    $args = @($ExtraArgs) + @("--measure-secs", "$MeasureSec", "--stats-file", $stats)
    $proc = Start-Process -FilePath $ExePath -ArgumentList $args -PassThru
    $samples = @()
    try {
        $cpuPrev = 0.0
        $t0 = [DateTime]::UtcNow
        while (-not $proc.HasExited -and ([DateTime]::UtcNow - $t0).TotalSeconds -lt ($MeasureSec + 8)) {
            Start-Sleep -Seconds 1
            $proc.Refresh()
            if ($proc.HasExited) { break }
            $cpu = $proc.TotalProcessorTime.TotalSeconds
            $samples += [PSCustomObject]@{
                T      = [Math]::Round(([DateTime]::UtcNow - $t0).TotalSeconds, 1)
                WsMB   = [Math]::Round($proc.WorkingSet64 / 1MB, 1)
                CpuPct = [Math]::Round((($cpu - $cpuPrev) / $cores) * 100.0, 2)
            }
            $cpuPrev = $cpu
        }
        if (-not $proc.WaitForExit(10000)) { $proc.Kill() }
    } finally {
        if (-not $proc.HasExited) { try { $proc.Kill() } catch {} }
    }
    $s = if (Test-Path $stats) { Get-Content $stats -Raw | ConvertFrom-Json } else { $null }
    $ws = ($samples | Measure-Object WsMB -Maximum -Average)
    [PSCustomObject]@{
        Run = $Label; Frames = $s.frames; AvgFps = $s.avg_fps; MinFps = $s.min_fps; P5Fps = $s.p5_fps
        WsPeakMB = $ws.Maximum; WsAvgMB = [Math]::Round($ws.Average, 1); WsEndMB = $s.working_set_mb
        Samples = $samples
    }
}

$streamSec = 22   # 2 s of lead-in + the 20 s script
$runs = @(
    (Measure-Run -Label "baseline-idle" -ExtraArgs @("--no-companion", "--force-idle") -MeasureSec $streamSec),
    (Measure-Run -Label "auto-no-companion" -ExtraArgs @("--no-companion") -MeasureSec $streamSec),
    (Measure-Run -Label "auto-companion-down" -ExtraArgs @() -MeasureSec $streamSec),
    (Measure-Run -Label "replay-20s-a" -ExtraArgs @("--companion-replay", $Replay) -MeasureSec $streamSec),
    (Measure-Run -Label "replay-20s-b" -ExtraArgs @("--companion-replay", $Replay) -MeasureSec $streamSec),
    (Measure-Run -Label "replay+tail" -ExtraArgs @("--companion-replay", $Replay) -MeasureSec ($streamSec + $TailSeconds))
)
$runs | Select-Object Run, Frames, AvgFps, MinFps, P5Fps, WsPeakMB, WsAvgMB, WsEndMB | Format-Table -AutoSize
foreach ($r in $runs) {
    "--- $($r.Run): per-second WS (MB) / CPU (% of all cores)"
    ($r.Samples | ForEach-Object { "{0,5}s  {1,7}  {2,6}" -f $_.T, $_.WsMB, $_.CpuPct }) -join "`n"
}
