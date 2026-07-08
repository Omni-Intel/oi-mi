# Unity Build Runtime

This project does not commit Unity build outputs to Git history. The Windows build is installed into a local ignored runtime directory:

```text
.runtime/unity/ARPrototype3D-windows-x64/ARPrototype3D.exe
```

## Fresh Windows Setup

After cloning or pulling the repository, run:

```bash
py -3.12 setup_local.py
source .venv/Scripts/activate
streamlit run gui.py
```

`setup_local.py` creates `.venv`, installs the Python package, downloads the Unity build zip, extracts it into `.runtime/`, and runs the environment check.

## GitHub Release Asset

Upload the Unity zip as a public GitHub Release asset named exactly:

```text
ARPrototype3D-windows-x64.zip
```

The default download URL is:

```text
https://github.com/Omni-Intel/oi-mi/releases/latest/download/ARPrototype3D-windows-x64.zip
```

## Local Zip Install

Before the GitHub Release exists, install from a local zip:

```bash
python tools/download_unity_build.py --from-local-zip /path/to/ARPrototype3D-windows-x64.zip
```

The zip must contain the full Unity Windows build folder contents, including:

```text
ARPrototype3D.exe
ARPrototype3D_Data/
UnityPlayer.dll
UnityCrashHandler64.exe
MonoBleedingEdge/
```

## Runtime Behavior

When `output.ar_game.enabled` and `output.ar_game.auto_launch` are both true, realtime decoding launches the configured executable if `127.0.0.1:5005` is not already listening.

The Unity process is not force-closed by default. This avoids killing a manually opened game during an experiment or while debugging.
