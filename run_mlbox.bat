@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  call install_and_run.bat
  exit /b
)

".venv\Scripts\python.exe" "%CD%\mlbox_app.py"
if errorlevel 1 (
  echo.
  echo WhiteML-Box failed to start. The virtual environment may be broken.
  echo Delete the .venv folder and run install_and_run.bat again.
  pause
  exit /b 1
)
