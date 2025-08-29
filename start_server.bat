@echo off
setlocal ENABLEDELAYEDEXPANSION
title Unity Backend Launcher

REM Detect venv Python
set PY_EXE="D:/AI MRO - PICO - AIRLINE/Backend/.venv/Scripts/python.exe"
if not exist %PY_EXE% (
    echo Could not find venv Python at %PY_EXE%.
    echo Please create/activate the virtual environment first.
    goto :end
)

REM Set the backend Python file here:
set SCRIPT=app_gry.py

echo ================================
echo   Unity Backend Launcher
echo ================================
echo.
echo Starting backend using %SCRIPT% ...
echo Flask server:  http://0.0.0.0:5000
echo WebSocket:     ws://0.0.0.0:8765
echo Press Ctrl+C to stop the server.
echo.

%PY_EXE% %SCRIPT%

:end
echo.
pause