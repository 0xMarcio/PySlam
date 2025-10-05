# Repository Guidelines

## Project Structure & Module Organization
- `slam.py` is the entry point; it streams frames from `videos/*.mp4`, manages the SLAM loop, and wires displays.
- Feature extraction, matching, and map primitives live in `frame.py`, `pointmap.py`, `helpers.py`, and `constants.py`; keep shared math utilities in `helpers.py` to avoid duplication.
- Visualization helpers reside in `display.py` and `renderer.py`, while `lib/{macosx,linux}` holds the prebuilt Pangolin/G2O extensions loaded at runtime.
- Support scripts, including ground-truth conversion, live under `tools/`, and sample pose files sit in `videos/groundtruth/`.

## Build, Test, and Development Commands
- Install Python dependencies via the system interpreter: `/opt/homebrew/bin/python3 -m pip install --break-system-packages -r requirements.txt`.
- Rebuild Pangolin and g2o only when you need new binaries—follow `INSTALL.md` and commit updated shared libraries under `lib/macosx/`.
- Run SLAM on the bundled sample clip: `DYLD_FALLBACK_LIBRARY_PATH="$PWD/lib/macosx:$DYLD_FALLBACK_LIBRARY_PATH" /opt/homebrew/bin/python3 slam.py videos/road.mp4`; pass a ground-truth pose file as a second argument to overlay evaluation.
- Convert KITTI/TUM truth data: `/opt/homebrew/bin/python3 tools/parse_ground_truth.py <input.txt> videos/groundtruth/<name>.npz`.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and snake_case for functions, variables, and filenames; keep class names in PascalCase (`Display3D`, `Frame`).
- Keep numpy arrays immutable where possible and prefer vectorized operations; limit module-level globals to configuration constants.
- Document non-obvious math or coordinate transforms with short comments placed above the block they describe.

## Testing Guidelines
- There is no automated suite yet; add `pytest` cases under `tests/` mirroring the module path (e.g., `tests/test_pointmap.py`).
- Name tests by behavior (`test_triangulate_handles_degenerate_pairs`) and seed deterministic random inputs for reproducibility.
- Before PRs, run `pytest` locally and exercise `python slam.py videos/test_drone.mp4` to confirm display and pose updates.

## Commit & Pull Request Guidelines
- Match the concise, imperative style already in history (`git log` shows entries like "code cleanup"); keep subjects ≤ 72 characters.
- Squash noisy work-in-progress commits before pushing and include co-authors in the trailer when applicable.
- PRs must describe the motivation, summarize algorithmic changes, list tested commands, and link related issues; attach screenshots or short clips when visual output changes.

## Runtime Configuration Tips
- `slam.py` reads environment variables (`F`, `SEEK`, `HEADLESS`) to adjust focal length, resume position, or disable the 2D view—document any new knobs.
- When adding native extensions, place platform-specific wheels under `lib/<platform>` and extend `sys.path` as done in the entry script.
