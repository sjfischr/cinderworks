@echo off
setlocal EnableDelayedExpansion

REM ============================================================================
REM  Cinderworks Studio — One-Click Bootstrap (Windows)
REM
REM  Checks prerequisites (Python 3.11, Git, uv), creates a project-local venv,
REM  installs exact-pinned dependencies, and launches the Gradio server.
REM
REM  If a prerequisite is missing, reports which one and exits non-zero
REM  WITHOUT creating or modifying the virtual environment.
REM
REM  This script does NOT perform git pull or self-directed pip install.
REM ============================================================================

REM --- Resolve paths relative to the studio root (parent of install/) ---
set "SCRIPT_DIR=%~dp0"
set "STUDIO_ROOT=%SCRIPT_DIR%.."
pushd "%STUDIO_ROOT%"
set "STUDIO_ROOT=%CD%"
popd

set "VENV_DIR=%STUDIO_ROOT%\.venv"
set "REQUIREMENTS=%STUDIO_ROOT%\requirements.txt"

REM ============================================================================
REM  Prerequisite checks — all must pass before any environment work
REM ============================================================================

set "MISSING="

REM --- Check Python >= 3.11 ---
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    set "MISSING=Python 3.11+"
    goto :report_missing
)

for /f "tokens=2 delims= " %%a in ('python --version 2^>^&1') do set "PY_VERSION=%%a"
for /f "tokens=1,2 delims=." %%x in ("!PY_VERSION!") do (
    set "PY_MAJOR=%%x"
    set "PY_MINOR=%%y"
)
if !PY_MAJOR! LSS 3 (
    set "MISSING=Python 3.11+ (found: !PY_VERSION!)"
    goto :report_missing
)
if !PY_MAJOR! EQU 3 if !PY_MINOR! LSS 11 (
    set "MISSING=Python 3.11+ (found: !PY_VERSION!)"
    goto :report_missing
)

REM --- Check Git ---
where git >nul 2>&1
if %ERRORLEVEL% neq 0 (
    set "MISSING=Git"
    goto :report_missing
)

REM --- Check uv ---
where uv >nul 2>&1
if %ERRORLEVEL% neq 0 (
    set "MISSING=uv"
    goto :report_missing
)

REM ============================================================================
REM  All prerequisites met — create venv and install
REM ============================================================================

echo [Cinderworks] All prerequisites found.
echo [Cinderworks] Creating project-local virtual environment...

uv venv "%VENV_DIR%" --python python
if %ERRORLEVEL% neq 0 (
    echo [Cinderworks] ERROR: Failed to create virtual environment.
    exit /b 1
)

echo [Cinderworks] Installing pinned dependencies from requirements.txt...

REM Install torch with CUDA support first (requires separate index)
echo [Cinderworks] Installing PyTorch with CUDA support...
uv pip install --python "%VENV_DIR%\Scripts\python.exe" torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
if %ERRORLEVEL% neq 0 (
    echo [Cinderworks] ERROR: Failed to install PyTorch with CUDA.
    exit /b 1
)

REM Install remaining dependencies (torch is already satisfied, will be skipped)
uv pip install --python "%VENV_DIR%\Scripts\python.exe" -r "%REQUIREMENTS%"
if %ERRORLEVEL% neq 0 (
    echo [Cinderworks] ERROR: Failed to install dependencies.
    exit /b 1
)

REM ============================================================================
REM  Launch the Gradio server
REM ============================================================================

echo [Cinderworks] Launching Cinderworks Studio...
"%VENV_DIR%\Scripts\python.exe" "%STUDIO_ROOT%\app.py"
exit /b %ERRORLEVEL%

REM ============================================================================
REM  Error reporting
REM ============================================================================

:report_missing
echo [Cinderworks] ERROR: Missing prerequisite: %MISSING%
echo [Cinderworks] Please install %MISSING% and try again.
exit /b 1
