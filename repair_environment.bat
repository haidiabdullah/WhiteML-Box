@echo off
setlocal
cd /d "%~dp0"

echo This will remove the existing WhiteML-Box virtual environment and rebuild it.
echo The app code and your data files will not be changed.
pause

if exist .venv (
  rmdir /s /q .venv
)

call install_and_run.bat
