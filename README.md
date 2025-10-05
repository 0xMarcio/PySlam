# PySlam

PySlam is a Python port of geohot's twitchslam with a refreshed Pangolin viewer and numerous practical fixes. The codebase focuses on readability and hackability while still exercising a real-time monocular SLAM pipeline.

## Prerequisites
- macOS Sonoma or newer with Apple Silicon (prebuilt `lib/macosx/*.dylib` are included)
- Python 3.13 (install via Homebrew and ensure `python3 --version` reports 3.13)
- Homebrew formulae: `cmake`, `eigen`, `suite-sparse`, `glew`, `glfw`, `libpng`, `jpeg`

## Quick Start
1. Clone the repository and `cd` into it.
2. Follow the step-by-step native build notes in [`INSTALL.md`](INSTALL.md) to install Pangolin (`pypangolin`) and g2o (`g2o.cpython-313-darwin.so`).
3. Install the Python requirements with:
   ```bash
   python3 -m pip install --break-system-packages -r requirements.txt
   ```
4. Run SLAM against the bundled 10-second sample clip (set `HEADLESS=1` to skip the Pangolin window when running remotely):
   ```bash
   HEADLESS=1 python3 slam.py videos/road.mp4
   ```
   Remove `HEADLESS=1` to open the Pangolin visualizer. The bundled loader primes the Pangolin `.dylib` search path automatically, so no manual `DYLD_*` overrides are required.

## Contributor Notes
- `AGENTS.md` documents repository layout, coding conventions, and PR etiquette.
- The bundled macOS binaries are rebuilt from source; update them whenever native dependencies change.
- Use Python 3.13 for development—older CPython builds cannot load the packaged native extensions.

For deeper architectural guidance or maintenance tips, see [`INSTALL.md`](INSTALL.md) and `AGENTS.md`.
