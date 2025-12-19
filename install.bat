@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM One-shot installer for Windows (venv + pip install).
REM Usage:
REM   install.bat
REM   install.bat --optional
REM   install.bat --dev
REM   install.bat --optional --dev

echo.
echo ============================================================
echo  AutoML with ClearML - Windows installer
echo ============================================================
echo.

set "VENV_DIR=.venv"
set "INSTALL_OPTIONAL=0"
set "INSTALL_DEV=0"

for %%A in (%*) do (
  if /I "%%~A"=="--optional" set "INSTALL_OPTIONAL=1"
  if /I "%%~A"=="--dev" set "INSTALL_DEV=1"
)

REM Resolve Python command (prefer py launcher).
set "PY_CMD="
where py >nul 2>nul
if not errorlevel 1 (
  set "PY_CMD=py -3"
) else (
  where python >nul 2>nul
  if not errorlevel 1 (
    set "PY_CMD=python"
  ) else (
    where python3 >nul 2>nul
    if not errorlevel 1 (
      set "PY_CMD=python3"
    )
  )
)

if "%PY_CMD%"=="" (
  echo [ERROR] Python was not found.
  echo         Install Python 3.10+ and ensure it is on PATH (recommended: install the Python Launcher 'py').
  exit /b 1
)

echo [INFO] Using Python: %PY_CMD%
%PY_CMD% -c "import sys; print(sys.version)" || goto :fail

REM Create venv if missing.
if not exist "%VENV_DIR%\\Scripts\\python.exe" (
  echo [INFO] Creating virtual environment: %VENV_DIR%
  %PY_CMD% -m venv "%VENV_DIR%" || goto :fail
)

REM Activate venv.
call "%VENV_DIR%\\Scripts\\activate.bat" || goto :fail

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip || goto :fail

echo [INFO] Installing requirements: requirements.txt
pip install -r requirements.txt || goto :fail

if "%INSTALL_OPTIONAL%"=="1" (
  if exist "requirements-optional.txt" (
    echo [INFO] Installing optional requirements: requirements-optional.txt
    pip install -r requirements-optional.txt || goto :fail
  ) else (
    echo [WARN] requirements-optional.txt not found; skipping.
  )
)

if "%INSTALL_DEV%"=="1" (
  if exist "requirements-dev.txt" (
    echo [INFO] Installing dev requirements: requirements-dev.txt
    pip install -r requirements-dev.txt || goto :fail
  ) else (
    echo [WARN] requirements-dev.txt not found; skipping.
  )
  REM pytest is not included in requirements-dev.txt; install it when dev tools requested.
  python -m pip show pytest >nul 2>nul
  if errorlevel 1 (
    echo [INFO] Installing pytest (dev)...
    pip install pytest || goto :fail
  )
)

REM Prepare ClearML config file (optional).
if not exist "clearml.conf" (
  if exist "clearml.conf.example" (
    copy /Y "clearml.conf.example" "clearml.conf" >nul
    echo [INFO] Created clearml.conf from clearml.conf.example (edit it with your server/credentials).
  )
)

echo.
echo [OK] Installation completed.
echo.
echo Next steps:
echo   1) Activate venv:
echo        call %VENV_DIR%\\Scripts\\activate.bat
echo   2) Run (examples):
echo        python -m automl_lib.cli.run_pipeline --config config.yaml
echo        python -m automl_lib.cli.run_training --config config_training.yaml
echo        python -m automl_lib.cli.run_inference --config inference_config.yaml
echo.
exit /b 0

:fail
echo.
echo [ERROR] Installation failed.
echo         If you see SSL/Proxy errors, configure your network/proxy and retry.
exit /b 1
