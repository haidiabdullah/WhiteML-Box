@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer from python.org and tick "Add Python to PATH".
    pause
    exit /b 1
  )
)

".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
  echo Pip upgrade failed.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m pip install -r "%CD%\requirements.txt"
if errorlevel 1 (
  echo Dependency installation failed.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --onefile --windowed --collect-all rasterio --add-data "white_ml_box_logo.png;." --name "WhiteML-Box" mlbox_app.py
if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)

echo.
echo Build complete. Your executable is: dist\WhiteML-Box.exe
echo To create a full Windows installer, install Inno Setup and compile WhiteML-Box.iss.
pause
