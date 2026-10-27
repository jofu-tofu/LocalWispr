$ErrorActionPreference = "Continue"
Set-Location "C:\EpicSource\Github\LocalWispr"
& ".\build.bat" stable
$exitCode = $LASTEXITCODE
Write-Host "Build exit code: $exitCode"
exit $exitCode
