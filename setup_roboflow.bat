@echo off
chcp 65001 >nul
echo ============================================================
echo   BuildingYOLO  -  Roboflow API Key Setup
echo   Dataset: utp-jtbn5/pipeline-tracks  (2553 images)
echo ============================================================
echo.
echo   HOW TO GET YOUR FREE API KEY:
echo   1. Go to https://app.roboflow.com  and sign up (free)
echo   2. Click your profile picture (top right)
echo   3. Click Settings
echo   4. Copy the Private API Key
echo   5. Come back here and right-click to paste it below
echo.
set /p RFKEY="Paste Roboflow API key and press Enter: "

if "%RFKEY%"=="" (
    echo No key entered. Exiting.
    pause
    exit /b 1
)

echo.
echo Testing API key...
python -c "from roboflow import Roboflow; rf=Roboflow(api_key='%RFKEY%'); print('  Connection OK')" 2>nul
if %errorlevel% neq 0 (
    echo.
    echo Invalid API key or network error.
    echo Please check your key at https://app.roboflow.com/settings/api
    pause
    exit /b 1
)

echo.
echo Saving to system environment...
setx ROBOFLOW_API_KEY "%RFKEY%"

echo.
echo ============================================================
echo   API key saved successfully.
echo.
echo   IMPORTANT: Close this window and open a NEW PowerShell.
echo   The key will be available in the new session.
echo.
echo   Then run:
echo     cd C:\path\to\BuildingYOLO
echo     python main.py
echo ============================================================
pause
