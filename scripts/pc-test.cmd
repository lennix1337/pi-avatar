@echo off
rem Roda os testes do pet no PC (a VM nao tem cargo). Uso: pc-task.mjs --name pet-test scripts/pc-test.cmd
cd /d C:\projetos\pi-avatar || exit /b 1
git fetch -q origin || exit /b 1
git merge --ff-only -q origin/main || exit /b 1
cd rune-pet || exit /b 1
cargo test --quiet
exit /b %ERRORLEVEL%
