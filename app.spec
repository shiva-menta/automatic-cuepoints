# -*- mode: python ; coding: utf-8 -*-
import re
import tomllib
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Parse dependencies from pyproject.toml to auto-generate hiddenimports
# SPECPATH is provided by PyInstaller and points to the directory containing the spec file
pyproject_path = Path(SPECPATH) / "pyproject.toml"
with open(pyproject_path, "rb") as f:
    pyproject = tomllib.load(f)

def parse_dependency(dep: str) -> str:
    """Extract package name from a dependency string.

    Handles:
    - Simple: pyrekordbox
    - Version specifiers: librosa>=0.10.0
    - Extras: audio-separator[cpu]
    - Git URLs: madmom @ git+https://...
    - Platform markers: pyqt5; sys_platform == 'darwin'
    """
    # Strip platform markers (everything after ;)
    dep = dep.split(";")[0].strip()
    # Strip git URL (everything after @)
    dep = dep.split("@")[0].strip()
    # Strip extras (everything in [])
    dep = re.sub(r"\[.*?\]", "", dep)
    # Strip version specifiers (>=, <=, <, >, ==, ~=, !=)
    dep = re.split(r"[<>=!~]", dep)[0].strip()
    # Normalize: replace - with _ for import compatibility
    return dep.replace("-", "_")

hiddenimports = [
    parse_dependency(dep) for dep in pyproject["project"]["dependencies"]
]

# Collect pyrekordbox package completely
pyrekordbox_datas, pyrekordbox_binaries, pyrekordbox_hiddenimports = collect_all('pyrekordbox')
hiddenimports += pyrekordbox_hiddenimports
hiddenimports += collect_submodules('pyrekordbox')

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=pyrekordbox_binaries,
    datas=[('Modal-IconMark.png', '.'), ('cuepoint_utils.py', '.'), ('track_interface', 'track_interface')] + pyrekordbox_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Autocuepoints',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Autocuepoints',
)
app = BUNDLE(
    coll,
    name='Autocuepoints.app',
    icon='icon.icns',
    bundle_identifier=None,
)
