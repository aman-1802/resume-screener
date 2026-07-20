@echo off
cd /d "D:\cv-screening-tool"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found at .venv\Scripts\python.exe
    echo The project may have moved or the venv was not set up correctly.
    pause
    exit /b 1
)

echo Starting CV Screening Tool...
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://127.0.0.1:8502"

".venv\Scripts\python.exe" "server.py"

echo.
echo The server has stopped. If that was unexpected, scroll up to see the error above.
pause
