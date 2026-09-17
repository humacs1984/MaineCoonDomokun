# -*- mode: python ; coding: utf-8 -*-
# MaineCoonDomokun macOS spec (onedir mode for .app bundle)
# Based on golden_vest_pet/GoldenVestPet.spec

a = Analysis(
    ['pet_engine.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets', 'assets'),
        ('icon.png', '.'),
    ],
    hiddenimports=['PyQt5.sip', 'objc', 'AppKit', 'Foundation'],
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Heavy third-party libs (safe to exclude)
        'numpy', 'scipy', 'matplotlib', 'pandas',
        'PIL', 'pillow',
        'cryptography', 'pytest', 'setuptools', 'pip', 'wheel',
        'tkinter',
        'transformers', 'torch', 'torchvision',
        'huggingface_hub', 'tokenizers', 'safetensors',
        'sam2', 'rembg', 'onnxruntime',
        'IPython', 'jupyter', 'notebook',
        'sympy', 'networkx',
        # Qt modules not used by pet (only Core+Gui+Widgets needed)
        'PyQt5.QtPdf', 'PyQt5.QtQuick', 'PyQt5.QtQml',
        'PyQt5.QtNetwork', 'PyQt5.QtVirtualKeyboard',
        'PyQt5.QtWebEngine', 'PyQt5.QtWebEngineCore',
        'PyQt5.QtWebEngineWidgets', 'PyQt5.QtWebKit',
        'PyQt5.QtXml', 'PyQt5.QtSvg', 'PyQt5.QtSql',
        'PyQt5.QtBluetooth', 'PyQt5.QtDBus',
        'PyQt5.QtDesigner', 'PyQt5.QtHelp',
        'PyQt5.QtMultimedia', 'PyQt5.QtMultimediaWidgets',
        'PyQt5.QtNfc', 'PyQt5.QtOpenGL',
        'PyQt5.QtPositioning', 'PyQt5.QtLocation',
        'PyQt5.QtPrintSupport', 'PyQt5.QtSensors',
        'PyQt5.QtSerialPort', 'PyQt5.QtTest',
        'PyQt5.QtTextToSpeech', 'PyQt5.QtXmlPatterns',
    ],
    noarchive=False,
    optimize=2,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MaineCoonDomokun',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='app.icns',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip_binaries=True,
    upx_exclude=[],
    name='MaineCoonDomokun',
)
