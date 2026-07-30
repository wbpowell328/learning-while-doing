@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo The demo has not been set up yet. Running setup...
    call setup-demo.bat
    if errorlevel 1 exit /b 1
)

if not exist "frontend\node_modules" (
    echo The dashboard has not been set up yet. Running setup...
    call setup-demo.bat
    if errorlevel 1 exit /b 1
)

echo Starting the API and dashboard in separate windows...
start "Learning While Doing API" /d "%~dp0" cmd /k ".venv\Scripts\python.exe -m uvicorn shell.app:app --host 127.0.0.1 --port 8000"
start "Learning While Doing Dashboard" /d "%~dp0frontend" cmd /k "npm.cmd run dev -- --host 127.0.0.1 --open"

echo.
echo The dashboard should open automatically at http://localhost:5173.
echo Leave the two new terminal windows open while using the demo.
pause
