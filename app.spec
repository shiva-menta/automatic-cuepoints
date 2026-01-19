# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[('Modal-IconMark.png', '.'), ('cuepoint_utils.py', '.'), ('track_interface', 'track_interface')],
    hiddenimports=[
        "sqlcipher3_wheels",
        "matplotlib",
        "librosa",
        "numpy",
        "scipy",
        "ruptures",
        "modal",
        "pyrekordbox",
    ],
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
