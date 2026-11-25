@echo off
rem Roda os testes do pet no PC (a VM nao tem cargo). Uso: npm test (pc-task.mjs --env PET_SHA scripts/pc-test.cmd)
rem Testa o commit PET_SHA (o HEAD da VM, ja enviado ao origin) num worktree proprio; sem PET_SHA, testa origin/main.
setlocal
set REPO=C:\projetos\pi-avatar
set WT=%REPO%\worktrees\verify
cd /d %REPO% || exit /b 1
git fetch -q origin || exit /b 1
if "%PET_SHA%"=="" (for /f %%S in ('git rev-parse origin/main') do set PET_SHA=%%S)
if not exist "%WT%\.git" (git worktree add -q --detach "%WT%" %PET_SHA% || exit /b 1)
cd /d "%WT%" || exit /b 1
git checkout -q --detach %PET_SHA% || exit /b 1
for /f %%S in ('git rev-parse HEAD') do echo testing %%S
if not exist rune-pet\.rune-cache mklink /J rune-pet\.rune-cache "%REPO%\rune-pet\.rune-cache" >nul || exit /b 1
set CARGO_TARGET_DIR=%REPO%\worktrees\verify-target
cd rune-pet || exit /b 1
cargo test --quiet
exit /b %ERRORLEVEL%
