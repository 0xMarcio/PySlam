# Installation Guide

This project ships with prebuilt macOS (arm64) binaries for Pangolin and g2o under `lib/macosx/`. Follow these steps if you need to reproduce the build or set up a fresh machine.

## 1. System Packages
Install the Homebrew prerequisites:

```bash
brew install cmake eigen suite-sparse glew glfw libpng jpeg
```

## 2. Python Dependencies
Use the Homebrew Python 3.13 interpreter and install runtime packages directly (no virtual environment):

```bash
/opt/homebrew/bin/python3 -m pip install --break-system-packages -r requirements.txt
```

## 3. Build Pangolin (pypangolin)
```bash
mkdir -p build-deps && cd build-deps
git clone https://github.com/stevenlovegrove/Pangolin.git
cmake -S Pangolin -B Pangolin/build -DBUILD_PANGOLIN_PYTHON=ON -DBUILD_EXAMPLES=OFF
cmake --build Pangolin/build -t pypangolin_wheel
/opt/homebrew/bin/python3 -m pip install --break-system-packages Pangolin/build/pypangolin-*.whl
```

## 4. Build g2o Python Bindings
```bash
cd build-deps
git clone https://github.com/uoip/g2opy.git
cd g2opy
# Refresh bundled pybind11 to a modern release
rm -rf EXTERNAL/pybind11
git clone --branch v2.13.6 https://github.com/pybind/pybind11.git EXTERNAL/pybind11
cmake -S . -B build \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DPYTHON_EXECUTABLE=/opt/homebrew/bin/python3 \
  -DBUILD_SHARED_LIBS=ON
cmake --build build -j4
cp build/lib/g2o.cpython-313-darwin.so ../../lib/macosx/
cp build/lib/libg2o_*.dylib ../../lib/macosx/
```

## 5. Verify the Installation
Run the sample clip with the freshly built libraries:

```bash
cd ../../
DYLD_FALLBACK_LIBRARY_PATH="$PWD/lib/macosx:$DYLD_FALLBACK_LIBRARY_PATH" \
HEADLESS=1 /opt/homebrew/bin/python3 slam.py videos/road.mp4
```

If you see frame-by-frame SLAM logs, Pangolin and g2o loaded correctly.

> Tip: Remove the temporary `build-deps/` directory after copying the shared objects into `lib/macosx/`.
