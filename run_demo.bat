@echo off
setlocal
cd /d "%~dp0"

set "CONDA="
set "LOCAL_CONDA_DIR=%LOCALAPPDATA%\slprt-miniconda3"
set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
set "MINICONDA_INSTALLER=%TEMP%\Miniconda3-latest-Windows-x86_64.exe"

for /f "delims=" %%I in ('where conda 2^>nul') do if not defined CONDA set "CONDA=%%I"
if defined CONDA goto :conda_ready

for %%I in ("%LOCAL_CONDA_DIR%\condabin\conda.bat" "%LOCAL_CONDA_DIR%\Scripts\conda.bat" "%LOCAL_CONDA_DIR%\Scripts\conda.exe" "%USERPROFILE%\miniconda3\condabin\conda.bat" "%USERPROFILE%\miniconda3\Scripts\conda.exe") do (
    if not defined CONDA if exist "%%~I" set "CONDA=%%~I"
)
if defined CONDA goto :conda_ready

echo Conda was not found. Installing Miniconda for this Windows user...
echo Downloading Miniconda from repo.anaconda.com...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri '%MINICONDA_URL%' -OutFile '%MINICONDA_INSTALLER%'"
if errorlevel 1 (
    echo Failed to download Miniconda from %MINICONDA_URL%.
    goto :failed
)

echo Installing Miniconda in %LOCAL_CONDA_DIR%...
start /wait "" "%MINICONDA_INSTALLER%" /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=%LOCAL_CONDA_DIR%
if errorlevel 1 (
    echo Miniconda installation failed.
    goto :failed
)
for %%I in ("%LOCAL_CONDA_DIR%\condabin\conda.bat" "%LOCAL_CONDA_DIR%\Scripts\conda.bat" "%LOCAL_CONDA_DIR%\Scripts\conda.exe" "%USERPROFILE%\miniconda3\condabin\conda.bat" "%USERPROFILE%\miniconda3\Scripts\conda.exe") do (
    if not defined CONDA if exist "%%~I" set "CONDA=%%~I"
)
if not defined CONDA (
    echo Miniconda installation completed, but no Conda executable was found under %LOCAL_CONDA_DIR% or %USERPROFILE%\miniconda3.
    goto :failed
)
call "%CONDA%" --version >nul 2>&1
if errorlevel 1 (
    echo Miniconda was found at %CONDA%, but it could not be started.
    goto :failed
)

:conda_ready
call "%CONDA%" env list | findstr /R /C:"^[ ]*slprt[ ]" >nul
if errorlevel 1 (
    echo Accepting Terms of Service for the required Anaconda channels...
    call "%CONDA%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    if errorlevel 1 goto :failed
    call "%CONDA%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    if errorlevel 1 goto :failed
    call "%CONDA%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
    if errorlevel 1 goto :failed

    echo Creating the slprt environment with Python 3.10.11...
    call "%CONDA%" create -n slprt python=3.10.11 -y
    if errorlevel 1 goto :failed
)

call "%CONDA%" run -n slprt python -c "import sys; assert sys.version_info[:3] == (3, 10, 11)" >nul 2>&1
if errorlevel 1 (
    echo The existing slprt environment must use Python 3.10.11.
    echo Rename or remove the incompatible environment, then run this file again.
    goto :failed
)

call "%CONDA%" run -n slprt python -c "import cv2, mediapipe, numpy, torch" >nul 2>&1
if errorlevel 1 (
    echo Installing demo requirements...
    call "%CONDA%" run -n slprt python -m pip install -r requirements.txt
    if errorlevel 1 goto :failed
)

echo Starting live handshape detection. Press Q in the display window to stop.
call "%CONDA%" run -n slprt python live_detect.py
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo Demo launch failed. Check the messages above.
pause
exit /b 1
