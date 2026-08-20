@echo off
setlocal
cd /d "%~dp0"

py -3.12 --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python 3.12 was not found. Run install_windows.bat first.
  pause
  exit /b 1
)

py -3.12 -c "import torch; print('[DEVICE] CUDA:', torch.cuda.is_available()); print('[DEVICE] GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
if errorlevel 1 (
  echo [ERROR] Required packages are missing. Run install_windows.bat first.
  pause
  exit /b 1
)
start "" http://127.0.0.1:7860
py -3.12 -m uvicorn app.main:app --host 127.0.0.1 --port 7860
pause
