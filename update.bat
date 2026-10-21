@echo off
REM Updates the CODE of this project from a new LegalDocAgent.zip without touching your .venv or .env
REM Usage:  update.bat "C:\path\to\LegalDocAgent.zip"
setlocal
if "%~1"=="" (
  echo Usage: update.bat "C:\path\to\LegalDocAgent.zip"
  exit /b 1
)
set "WORK=%TEMP%\lda_update"
rmdir /s /q "%WORK%" 2>nul
powershell -NoProfile -Command "Expand-Archive -LiteralPath '%~1' -DestinationPath '%WORK%' -Force"
if errorlevel 1 (
  echo Could not unzip the file.
  exit /b 1
)
robocopy "%WORK%\LegalDocAgent" "%~dp0." /E /XD .venv .git __pycache__ reports /XF .env update.bat >nul
rmdir /s /q "%WORK%" 2>nul
echo.
echo Done. Your .venv and .env were NOT touched. Start the app with start.bat
endlocal
exit /b 0
