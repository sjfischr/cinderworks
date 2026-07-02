@echo off
REM Cinderworks Studio — Quick Launch (Windows)
REM Activates the venv and starts the Gradio server.

set "STUDIO_ROOT=%~dp0"
set "VENV_DIR=%STUDIO_ROOT%.venv"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [Cinderworks] No .venv found. Run install\bootstrap.bat first.
    exit /b 1
)

echo [Cinderworks] Launching...
"%VENV_DIR%\Scripts\python.exe" "%STUDIO_ROOT%app.py"
