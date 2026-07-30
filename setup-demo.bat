@echo off
setlocal

cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python 3.11 or newer is required. Install it from https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python environment...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo Could not create the Python environment. Confirm Python 3.11 is installed.
        pause
        exit /b 1
    )
)

echo Installing Python packages...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo Python package installation failed.
    pause
    exit /b 1
)

where npm.cmd >nul 2>nul
if errorlevel 1 (
    echo Node.js 20 or newer is required. Install it from https://nodejs.org/
    pause
    exit /b 1
)

if not exist "frontend\node_modules" (
    echo Installing the dashboard packages...
    pushd frontend
    call npm.cmd ci
    popd
    if errorlevel 1 (
        echo Dashboard package installation failed.
        pause
        exit /b 1
    )
)

echo.
echo Setup complete. Run start-demo.bat to launch the interactive dashboard.
pause
