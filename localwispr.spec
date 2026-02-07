# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for LocalWispr.

Build with:
    pyinstaller localwispr.spec

Or use:
    build.bat           - Build stable version
    build.bat test      - Build test version
    build.bat both      - Build both versions
"""

import os
import sys
from pathlib import Path

# Build variant parameterization
BUILD_VARIANT = os.environ.get("BUILD_VARIANT", "stable")
APP_NAME = "LocalWispr-Test" if BUILD_VARIANT == "test" else "LocalWispr"
SHOW_CONSOLE = BUILD_VARIANT == "test"  # Test shows console, stable hides it

block_cipher = None

# Project root
ROOT = Path(SPECPATH)

# Collect data files
datas = [
    # Prompt templates
    (str(ROOT / 'localwispr' / 'prompts' / '*.txt'), 'localwispr/prompts'),
]

# Collect pywhispercpp data files if present
from PyInstaller.utils.hooks import collect_data_files
try:
    datas += collect_data_files('pywhispercpp')
except Exception:
    pass

# Collect setuptools/jaraco data files (needed for pkg_resources)
try:
    datas += collect_data_files('setuptools._vendor.jaraco.text')
except Exception:
    pass

# Hidden imports that PyInstaller might miss
hiddenimports = [
    # pystray Windows backend
    'pystray._win32',
    # pynput backends
    'pynput.keyboard._win32',
    'pynput.mouse._win32',
    # pywhispercpp dependencies
    'pywhispercpp',
    'pywhispercpp.model',
    # Audio
    'sounddevice',
    '_sounddevice_data',
    # Windows COM for pycaw
    'comtypes',
    'comtypes.client',
    'comtypes.stream',
    # winotify
    'winotify',
    # PIL/Pillow
    'PIL._tkinter_finder',
    # torch for Silero VAD
    'torch',
    'torchaudio',
]

a = Analysis(
    [str(ROOT / 'localwispr' / '__main__.py')],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        # Note: tkinter is required by overlay.py
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,  # Creates folder bundle (faster startup)
    name=APP_NAME,
    debug=True,  # Enable debug for troubleshooting
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=SHOW_CONSOLE,  # Test shows console, stable hides it
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='assets/localwispr.ico',  # Uncomment when icon is added
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)
