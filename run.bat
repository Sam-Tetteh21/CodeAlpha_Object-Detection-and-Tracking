@echo off
setlocal
cd /d "%~dp0"

echo Starting CodeAlpha Object Detection and Tracking...

if exist "venv\Scripts\python.exe" (
    set "PYTHON=venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON=.venv\Scripts\python.exe"
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON=py -3"
    ) else (
        where python >nul 2>nul
        if errorlevel 1 (
            echo Python was not found. Install Python 3 and try again.
            pause
            exit /b 1
        )
        set "PYTHON=python"
    )
)

if not exist "yolov4-tiny.weights" (
    echo Missing model file: yolov4-tiny.weights
    echo Keep yolov4-tiny.weights, yolov4-tiny.cfg, and coco.names beside app.py.
    pause
    exit /b 1
)

%PYTHON% -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo The app could not be started. Install dependencies with:
    echo %PYTHON% -m pip install -r requirements.txt
    pause
)

endlocal
