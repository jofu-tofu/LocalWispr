@echo off
REM LocalWispr Dual-Version Build Script
REM Usage:
REM   build.bat           - Build stable version (hidden console)
REM   build.bat stable    - Build stable version (hidden console)
REM   build.bat test      - Build test version (visible console)
REM   build.bat both      - Build both versions
REM   build.bat installer - Build stable + Windows installer

setlocal enabledelayedexpansion

REM Parse argument
set "BUILD_TARGET=%~1"
if "%BUILD_TARGET%"=="" set "BUILD_TARGET=stable"

echo ========================================
echo LocalWispr Build System
echo Target: %BUILD_TARGET%
echo ========================================

REM Use venv Python
set PYTHON=.venv\Scripts\python.exe
if not exist %PYTHON% (
    echo Error: Virtual environment not found at .venv
    echo Run: uv venv .venv ^&^& uv pip install -e .
    exit /b 1
)

REM Check if pytest is installed
%PYTHON% -c "import pytest" 2>nul
if errorlevel 1 (
    echo Installing pytest and test dependencies...
    uv pip install pytest pytest-mock pytest-cov pytest-xdist
)

REM Check if pyinstaller is installed
%PYTHON% -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing pyinstaller...
    uv pip install pyinstaller>=6.0
)

echo.
echo [1/3] Running unit and integration tests...
echo ----------------------------------------
%PYTHON% -m pytest tests/ -v --ignore=tests/gui -x
if errorlevel 1 (
    echo.
    echo ========================================
    echo TESTS FAILED - Build aborted.
    echo ========================================
    echo.
    echo Fix the failing tests before building.
    echo.
    echo Tips:
    echo   - Run "pytest tests/ -v" to see all failures
    echo   - Run "pytest tests/test_MODULE.py -v" to test specific module
    echo   - GUI tests are excluded from build gate (run manually with: pytest tests/gui/ -v)
    echo.
    exit /b 1
)

echo.
echo ----------------------------------------
echo Tests passed!
echo ----------------------------------------

REM Route to appropriate build function
if /i "%BUILD_TARGET%"=="both" (
    call :build_variant stable
    if errorlevel 1 exit /b 1
    call :build_variant test
    if errorlevel 1 exit /b 1
    goto :build_complete_both
) else if /i "%BUILD_TARGET%"=="test" (
    call :build_variant test
    if errorlevel 1 exit /b 1
    goto :build_complete_test
) else if /i "%BUILD_TARGET%"=="installer" (
    call :build_variant stable
    if errorlevel 1 exit /b 1
    call build-installer.bat
    if errorlevel 1 exit /b 1
    goto :build_complete_installer
) else (
    call :build_variant stable
    if errorlevel 1 exit /b 1
    goto :build_complete_stable
)

:build_variant
REM Build a specific variant
REM %1 = variant name (stable or test)
set "VARIANT=%~1"
echo.
echo ========================================
echo Building %VARIANT% version...
echo ========================================

REM Set environment variable for spec file
set "BUILD_VARIANT=%VARIANT%"

REM Determine output folder and config file
if /i "%VARIANT%"=="test" (
    set "OUTPUT_FOLDER=LocalWispr-Test"
    set "CONFIG_FILE=config-test.toml"
    set "EXE_NAME=LocalWispr-Test.exe"
) else (
    set "OUTPUT_FOLDER=LocalWispr"
    set "CONFIG_FILE=config.toml"
    set "EXE_NAME=LocalWispr.exe"
)

REM Clean previous build for this variant only
if exist "dist\%OUTPUT_FOLDER%" (
    echo Cleaning previous %VARIANT% build...
    rmdir /s /q "dist\%OUTPUT_FOLDER%"
)

echo.
echo [2/3] Building EXE (%VARIANT%)...
echo ----------------------------------------
%PYTHON% -m PyInstaller --noconfirm localwispr.spec
if errorlevel 1 (
    echo.
    echo ========================================
    echo BUILD FAILED (%VARIANT%)
    echo ========================================
    exit /b 1
)

REM Copy config file to dist folder as config-defaults.toml
echo.
echo [3/3] Copying config (%VARIANT%)...
if exist "dist\%OUTPUT_FOLDER%" (
    copy "%CONFIG_FILE%" "dist\%OUTPUT_FOLDER%\config-defaults.toml" >nul
    if errorlevel 1 (
        echo Warning: Failed to copy %CONFIG_FILE%
    )
)

echo %VARIANT% build complete!
exit /b 0

:build_complete_stable
echo.
echo ========================================
echo STABLE Build complete!
echo Output: dist\LocalWispr\LocalWispr.exe
echo Console: Hidden
echo Hotkey: Win+Ctrl+Shift
echo.
echo To run: dist\LocalWispr\LocalWispr.exe
echo ========================================
goto :eof

:build_complete_test
echo.
echo ========================================
echo TEST Build complete!
echo Output: dist\LocalWispr-Test\LocalWispr-Test.exe
echo Console: Visible (for debugging)
echo Hotkey: Ctrl+Alt+Shift
echo.
echo To run: dist\LocalWispr-Test\LocalWispr-Test.exe
echo ========================================
goto :eof

:build_complete_both
echo.
echo ========================================
echo BOTH Builds complete!
echo.
echo STABLE: dist\LocalWispr\LocalWispr.exe
echo   - Console: Hidden
echo   - Hotkey: Win+Ctrl+Shift
echo   - Icon: Blue wave
echo.
echo TEST: dist\LocalWispr-Test\LocalWispr-Test.exe
echo   - Console: Visible
echo   - Hotkey: Ctrl+Alt+Shift
echo   - Icon: Orange wave
echo.
echo Both can run simultaneously!
echo ========================================
goto :eof

:build_complete_installer
echo.
echo ========================================
echo INSTALLER Build complete!
echo.
echo Installer: dist\LocalWispr-Setup-0.1.0.exe
echo.
echo To test:
echo   1. Run the installer
echo   2. Install to a clean location
echo   3. First-run wizard will download a model
echo   4. Test transcription with hotkey
echo ========================================
goto :eof
