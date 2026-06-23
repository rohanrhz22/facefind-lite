@echo off
REM FaceFind Lite - one-click start (Windows)
cd /d "%~dp0backend"
python -m pip install -r requirements.txt
echo.
echo Starting FaceFind Lite at http://127.0.0.1:8000
python -m uvicorn app:app --host 127.0.0.1 --port 8000
