@echo off

REM Check if Python is installed
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo Python could not be found. Please install it and try again.
    exit /b
)

REM Check if pip is installed
pip --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo pip could not be found. Please install it and try again.
    exit /b
)

REM Install the required Python packages
pip install -r requirements.txt

REM Run the game
python q_gui.py