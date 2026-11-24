param(
    [string]$Rev = "2312d60bf07a300a3142dace9d067b24f2454488",
    [string]$TargetDir = "$PSScriptRoot\..\.rune-cache"
)

$ErrorActionPreference = 'Stop'

if (Test-Path (Join-Path $TargetDir "crates\rune\Cargo.toml")) {
    Write-Host "Rune sparse cache already present at $TargetDir"
    exit 0
}

Remove-Item -Recurse -Force $TargetDir -ErrorAction SilentlyContinue
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/anomalyco/rune.git $TargetDir
git -C $TargetDir sparse-checkout init --cone
git -C $TargetDir sparse-checkout set `
    crates/rune `
    crates/rune-core `
    crates/rune-edit `
    crates/rune-gpu `
    crates/rune-image `
    crates/rune-media `
    crates/rune-motion `
    crates/rune-spell `
    crates/rune-staging `
    crates/rune-taffy `
    crates/rune-text `
    vendor/wgpu-hal `
    vendor/naga `
    vendor/winit `
    vendor/parley_engine `
    vendor/sctk-adwaita `
    vendor/tiny-skia-path
git -C $TargetDir fetch --depth 1 origin $Rev
if ($LASTEXITCODE -ne 0) { throw 'Could not fetch the pinned Rune revision' }
git -C $TargetDir checkout $Rev
if ($LASTEXITCODE -ne 0) { throw 'Could not check out the pinned Rune revision' }
Write-Host "Rune sparse cache ready at $TargetDir ($Rev)"
