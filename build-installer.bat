@echo off
REM LocalWispr Installer Build Script
REM Builds the Windows installer using Inno Setup
REM
REM Prerequisites:
REM   1. Install Inno Setup 6 from https://jrsoftware.org/isdl.php
REM   2. Run "build.bat stable" first to create the EXE

setlocal enabledelayedexpansion

echo ========================================
echo LocalWispr Installer Build
echo ========================================

REM Check for Inno Setup
set "ISCC="

REM Try common installation paths
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
)
if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
    set "ISCC=C:\Program Files\Inno Setup 6\ISCC.exe"
)

REM Check if ISCC was found
if "!ISCC!"=="" (
    echo Error: Inno Setup 6 not found.
    echo.
    echo Please install Inno Setup 6 from:
    echo   https://jrsoftware.org/isdl.php
    echo.
    echo Expected installation paths:
    echo   C:\Program Files (x86)\Inno Setup 6\ISCC.exe
    echo   C:\Program Files\Inno Setup 6\ISCC.exe
    exit /b 1
)

echo Found Inno Setup at: !ISCC!
echo.

REM Check for stable build
if not exist "dist\LocalWispr\LocalWispr.exe" (
    echo Error: Stable build not found at dist\LocalWispr\LocalWispr.exe
    echo.
    echo Run "build.bat stable" first to create the EXE.
    exit /b 1
)

REM Check for config-defaults.toml
if not exist "dist\LocalWispr\config-defaults.toml" (
    echo Warning: config-defaults.toml not found, copying from config.toml...
    if exist "config.toml" (
        copy "config.toml" "dist\LocalWispr\config-defaults.toml" >nul
    ) else (
        echo Error: No config file found.
        exit /b 1
    )
)

echo Building installer...
echo.

REM Run Inno Setup compiler
"!ISCC!" installer\localwispr.iss

if errorlevel 1 (
    echo.
    echo ========================================
    echo INSTALLER BUILD FAILED
    echo ========================================
    exit /b 1
)

REM Extract version from ISS file for display
for /f "tokens=2 delims==" %%a in ('findstr /C:"#define MyAppVersion" installer\localwispr.iss') do (
    set "ISS_VERSION=%%a"
)
REM Remove quotes and spaces
set "ISS_VERSION=!ISS_VERSION:"=!"
set "ISS_VERSION=!ISS_VERSION: =!"

echo.
echo ========================================
echo INSTALLER BUILD COMPLETE
echo ========================================
echo.
echo Output: dist\LocalWispr-Setup-!ISS_VERSION!.exe
echo.
echo Test the installer:
echo   1. Run dist\LocalWispr-Setup-!ISS_VERSION!.exe
echo   2. Install to a clean location
echo   3. Verify first-run wizard appears
echo   4. Download a model and test transcription
echo.
