$ErrorActionPreference = "Stop"
Set-Location "C:\EpicSource\Github\LocalWispr"

Write-Host "========================================"
Write-Host "Building LocalWispr Stable"
Write-Host "========================================"

# Set environment variable
$env:BUILD_VARIANT = "stable"

# Clean previous build
if (Test-Path "dist\LocalWispr") {
    Write-Host "Cleaning previous stable build..."
    Remove-Item -Recurse -Force "dist\LocalWispr"
}

Write-Host ""
Write-Host "[2/3] Building EXE (stable)..."
Write-Host "----------------------------------------"
& ".venv\Scripts\python.exe" -m PyInstaller --noconfirm localwispr.spec
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "BUILD FAILED (stable)"
    Write-Host "========================================"
    exit 1
}

Write-Host ""
Write-Host "[3/3] Copying config (stable)..."
if (Test-Path "dist\LocalWispr") {
    Copy-Item "config.toml" "dist\LocalWispr\config-defaults.toml"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Warning: Failed to copy config.toml"
    }
}

Write-Host ""
Write-Host "========================================"
Write-Host "STABLE Build complete!"
Write-Host "Output: dist\LocalWispr\LocalWispr.exe"
Write-Host "Console: Hidden"
Write-Host "Hotkey: Win+Ctrl+Shift"
Write-Host ""
Write-Host "To run: dist\LocalWispr\LocalWispr.exe"
Write-Host "========================================"
