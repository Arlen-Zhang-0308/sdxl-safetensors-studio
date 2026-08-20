@echo off
setlocal
cd /d "%~dp0"

py -3.12 --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python 3.12 was not found.
  echo Install it with: winget install Python.Python.3.12
  pause
  exit /b 1
)

py -3.12 -m pip install --upgrade pip setuptools wheel

echo [INFO] Installing CUDA-enabled PyTorch first...
py -3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
  echo [WARN] CUDA PyTorch install failed. Falling back to CPU build.
  py -3.12 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  if errorlevel 1 goto :failed
)

py -3.12 -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo.
echo [OK] Installation complete. Put SDXL files in models, LoRA files in loras,
echo      and complete local IP-Adapter packages in ip_adapters.
pause
exit /b 0

:failed
echo [ERROR] Installation failed. Check your network and Python installation.
pause
exit /b 1
