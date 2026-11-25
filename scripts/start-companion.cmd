@echo off
rem Sobe a companheira: o pet (rune-pet) e o companion (cerebro Pi + voz real). Ctrl+C encerra os dois.
rem Uso: scripts\start-companion.cmd [--text] [--model anthropic/claude-haiku-5-5] [--no-pet] [--doctor: so confere o ambiente, sem abrir o pet] [...flags do companion]
rem Sem microfone/voz instalada: --text conversa pelo teclado. Primeira vez: baixa as dependencias (uv) e compila o pet.
setlocal
set "ROOT=%~dp0.."
set "PET=%ROOT%\rune-pet\target\release\pi-avatar.exe"
set "STARTED_PET="
set "ARGS="
set "WANT_PET=1"
:args
if "%~1"=="" goto run
if /i "%~1"=="--no-pet" (set "WANT_PET=") else (set ARGS=%ARGS% %1)
if /i "%~1"=="--doctor" set "WANT_PET="
shift
goto args

:run
if defined WANT_PET (
  if not exist "%PET%" (
    echo [start] compilando o pet pela primeira vez...
    pushd "%ROOT%\rune-pet" || exit /b 1
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup-rune.ps1 || exit /b 1
    cargo build --release || exit /b 1
    popd
  )
  tasklist /fi "imagename eq pi-avatar.exe" | find /i "pi-avatar.exe" >nul
  if errorlevel 1 (
    echo [start] abrindo o pet
    start "" "%PET%" --companion 127.0.0.1:47800
    set "STARTED_PET=1"
  ) else (
    echo [start] o pet ja esta aberto; ele reconecta sozinho ao companion
  )
)

cd /d "%ROOT%\companion" || exit /b 1
echo [start] companion (cerebro + voz) - Ctrl+C para sair
uv run --extra voice python -m companion %ARGS%
set "RC=%ERRORLEVEL%"

if defined STARTED_PET taskkill /im pi-avatar.exe /f >nul 2>&1
exit /b %RC%
