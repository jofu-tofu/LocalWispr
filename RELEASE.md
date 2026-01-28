# Release Checklist

This document outlines the process for creating a new LocalWispr release.

## Pre-Release

### 1. Verify Code Quality
- [ ] All tests pass: `uv run pytest tests/ -v --ignore=tests/gui`
- [ ] No known critical bugs or regressions
- [ ] Test build works locally: `build.bat test`
- [ ] Manual testing of key features:
  - [ ] Hotkey recording and transcription
  - [ ] Model download in Settings
  - [ ] Streaming mode (if enabled)
  - [ ] Settings persistence
  - [ ] Tray icon and menu

### 2. Update Version Numbers

**IMPORTANT**: Version must be updated in **all three files** or the build will fail.

- [ ] `localwispr/__init__.py` - Update `__version__ = "X.X.X"`
- [ ] `pyproject.toml` - Update `version = "X.X.X"`
- [ ] `installer/localwispr.iss` - Update `#define MyAppVersion "X.X.X"`

**Verify all three match:**
```powershell
# Quick verification script
$pyVer = (Get-Content localwispr\__init__.py | Select-String '__version__').ToString().Split('"')[1]
$tomlVer = (Get-Content pyproject.toml | Select-String 'version =').ToString().Split('"')[1]
$issVer = (Get-Content installer\localwispr.iss | Select-String 'MyAppVersion').ToString().Split('"')[1]

if ($pyVer -eq $tomlVer -and $tomlVer -eq $issVer) {
    Write-Host "✓ All versions match: $pyVer" -ForegroundColor Green
} else {
    Write-Host "✗ Version mismatch!" -ForegroundColor Red
    Write-Host "  __init__.py: $pyVer"
    Write-Host "  pyproject.toml: $tomlVer"
    Write-Host "  localwispr.iss: $issVer"
}
```

### 3. Local Build Test

Test the complete build and installer process locally:

```bash
# Build stable EXE
build.bat stable

# Build installer
build-installer.bat
```

Expected output:
- `dist\LocalWispr\LocalWispr.exe` created
- `dist\LocalWispr-Setup-X.X.X.exe` created

**Manual install test:**
- [ ] Run the installer on your development machine
- [ ] Verify app launches and tray icon appears
- [ ] Test model download in Settings
- [ ] Test recording and transcription
- [ ] Uninstall cleanly

## Create Release

### 1. Commit Version Changes

```bash
# Stage version changes
git add localwispr/__init__.py pyproject.toml installer/localwispr.iss

# Commit with version number
git commit -m "Release v0.X.X"

# Push to master
git push origin master
```

### 2. Create and Push Tag

```bash
# Create annotated tag (use the same version as in files)
git tag -a v0.X.X -m "Release v0.X.X"

# Push tag to trigger GitHub Actions
git push origin v0.X.X
```

### 3. Monitor GitHub Actions

1. Go to: https://github.com/jofu-tofu/LocalWispr/actions
2. Watch the "Build and Release" workflow
3. Expected steps:
   - ✓ Checkout code
   - ✓ Set up Python
   - ✓ Install UV
   - ✓ Install dependencies
   - ✓ Run tests
   - ✓ Build stable EXE
   - ✓ Install Inno Setup
   - ✓ Build installer
   - ✓ Create GitHub Release

**If workflow fails:**
- Check the logs for the failing step
- Fix the issue locally
- Delete the tag: `git tag -d v0.X.X && git push origin :v0.X.X`
- Delete the draft release on GitHub (if created)
- Increment to a patch version (e.g., v0.X.Y)
- Start over from Pre-Release checklist

## Post-Release

### 1. Verify GitHub Release

- [ ] Release appears at: https://github.com/jofu-tofu/LocalWispr/releases
- [ ] Release has the correct version number
- [ ] `LocalWispr-Setup-X.X.X.exe` is attached
- [ ] Release notes are auto-generated and make sense
- [ ] Release is marked as "Latest"

### 2. Test Downloaded Installer

**Ideal test (if available):** Clean Windows VM or spare PC without Python/UV

- [ ] Download installer from GitHub release page
- [ ] Run installer
- [ ] Verify installation completes without errors
- [ ] Test app launches and works
- [ ] Test model download
- [ ] Test transcription
- [ ] Test uninstaller

**Minimum test:** Download on development machine and verify it runs

### 3. Update Documentation (if needed)

- [ ] Check if README needs updates
- [ ] Check if INSTALLATION.md needs updates
- [ ] Update any version-specific documentation

### 4. Announce Release (optional)

- [ ] Post release notes to relevant communities
- [ ] Update any external documentation
- [ ] Notify beta testers

## Rollback Procedure

If a critical bug is discovered after release:

### Option 1: Hotfix Release (Preferred)

1. Fix the bug on master
2. Update version to patch release (e.g., v0.1.0 → v0.1.1)
3. Follow normal release process above
4. Mark the broken release as "pre-release" on GitHub (edit release)

### Option 2: Delete Release (Last Resort)

Only use if the release is completely broken and unusable.

1. Delete the GitHub release (Releases page → Edit → Delete)
2. Delete the tag locally: `git tag -d v0.X.X`
3. Delete the tag remotely: `git push origin :v0.X.X`
4. Fix the issue
5. Increment version and start over

**Note:** Deleting releases should be rare - prefer hotfix releases.

## Version Numbering Guide

LocalWispr uses semantic versioning: `MAJOR.MINOR.PATCH`

- **MAJOR** (0.X.X): Breaking changes, major rewrites
- **MINOR** (X.1.X): New features, significant improvements
- **PATCH** (X.X.1): Bug fixes, small improvements

Examples:
- `v0.1.0` → `v0.1.1`: Bug fix release
- `v0.1.0` → `v0.2.0`: Added new feature (e.g., streaming mode)
- `v0.9.0` → `v1.0.0`: First stable release

## Troubleshooting

### "Version mismatch" error in build
- All three version files must match exactly
- Re-run the verification script above

### GitHub Actions fails at "Run tests"
- Tests are failing - fix them locally first
- Run `uv run pytest tests/ -v --ignore=tests/gui` before releasing

### GitHub Actions fails at "Build installer"
- Check if Inno Setup installed correctly in the workflow
- Verify `build-installer.bat` works locally

### Installer not appearing in release
- Check if the installer filename matches the pattern in workflow
- Verify build-installer.bat created the file in `dist/`

### Release notes are empty
- GitHub couldn't find commits since last tag
- Manually edit the release to add notes
