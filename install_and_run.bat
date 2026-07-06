@echo off
setlocal
cd /d "%~dp0"

echo Starting WhiteML-Box setup...
echo Project folder: %CD%

REM Repair case: old virtual environments can break after the folder is moved/renamed.
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -c "import sys" >nul 2>&1
  if errorlevel 1 (
    echo Existing virtual environment is broken. Recreating it...
    rmdir /s /q .venv
  )
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating Python virtual environment for WhiteML-Box...
  py -3 -m venv .venv
  if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer from python.org and tick "Add Python to PATH".
    pause
    exit /b 1
  )
)

echo Installing required packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
  echo Pip upgrade failed.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m pip install -r "%CD%\requirements.txt"
if errorlevel 1 (
  echo Dependency installation failed.
  echo.
  echo Try deleting the .venv folder, then run this file again.
  pause
  exit /b 1
)

echo Launching WhiteML-Box...
".venv\Scripts\python.exe" "%CD%\mlbox_app.py"
pause
