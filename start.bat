@echo off
REM Starts the app using the existing .venv (no installing, no re-creating anything)
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
streamlit run app.py
