# PySlam

PySlam is a Python port of geohot's twitchslam with a refreshed Pangolin viewer and numerous practical fixes. The codebase focuses on readability and hackability while still exercising a real-time monocular SLAM pipeline.

## Prerequisites
- macOS Sonoma or newer with Apple Silicon (prebuilt `lib/macosx/*.dylib` are included)
- Python 3.13 from Homebrew (`/opt/homebrew/bin/python3`)
- Homebrew formulae: `cmake`, `eigen`, `suite-sparse`, `glew`, `glfw`, `libpng`, `jpeg`

## Quick Start
1. Clone the repository and `cd` into it.
2. Follow the step-by-step native build notes in [`INSTALL.md`](INSTALL.md) to install Pangolin (`pypangolin`) and g2o (`g2o.cpython-313-darwin.so`).
3. Install the Python requirements with:
   ```bash
   /opt/homebrew/bin/python3 -m pip install --break-system-packages -r requirements.txt
   ```
4. Run SLAM against a bundled sample clip:
   ```bash
   DYLD_FALLBACK_LIBRARY_PATH="$PWD/lib/macosx:$DYLD_FALLBACK_LIBRARY_PATH" \
   HEADLESS=1 /opt/homebrew/bin/python3 slam.py videos/test_road.mp4
   ```
   Remove `HEADLESS=1` to open the Pangolin visualizer.

## Contributor Notes
- `AGENTS.md` documents repository layout, coding conventions, and PR etiquette.
- The bundled macOS binaries are rebuilt from source; update them whenever native dependencies change.
- Use Python 3.13 for development—older CPython builds cannot load the packaged native extensions.

For deeper architectural guidance or maintenance tips, see [`INSTALL.md`](INSTALL.md) and `AGENTS.md`.
