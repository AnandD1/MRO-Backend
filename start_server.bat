@echo off
echo Starting Unity WebRTC Backend Server...
echo.
echo Server will start on http://0.0.0.0:8080
echo WebSocket endpoint: ws://10.214.140.132:8080/ws
echo Network accessible from Pico at: ws://10.214.140.132:8080/ws
echo.
echo Press Ctrl+C to stop the server
echo Press 'q' in the video window to close it
echo.

"D:/AI MRO - PICO - AIRLINE/Backend/.venv/Scripts/python.exe" webrtc_server.py

pause
