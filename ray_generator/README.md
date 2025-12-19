# Ray Generator

Please refer to `examples/mesh_sim.py` and `tests/rir_test.py` for usage on dumping the raw path!
Please follow this compiling guide to install, instead of installing from pip.

## Path Data Keys

The dump path keys are:
- `source_indices`: Which sound source each path originated from
- `path_types`: Path type flags where DIRECT=1 (straight path), TRANSMISSION=2 (through materials), SPECULAR=4 (mirror-like reflection), DIFFUSE=8 (scattered reflection), DIFFRACTION=16 (around obstacles)
- `distances`: The total distance traveled by each path
- `listener_directions`: The arrival direction vector (xyz) at the listener for each path
- `source_directions`: The departure direction vector (xyz) from the source for each path
- `relative_speeds`: The relative velocity between source and listener for each path
- `speeds_of_sound`: The speed of sound along each path
- `intensities`: The sound intensity values across 8 frequency bands for each path
- `num_paths`: Total number of paths
- `num_bands`: Number of frequency bands analyzed

Frequency bands are as defined in `src/pygsound/src/Context.cpp`. Change as needed.
See `pipeline.py` for a pipeline generating ray SIRs.

---

## Dependencies

### Linux
```bash
sudo apt-get update
sudo apt-get -y install libfftw3-dev python3-dev zlib1g-dev
```

### MacOS
```bash
brew update
brew install fftw
```

---

## Installation (CPU Only)

```bash
git clone --recurse-submodules https://github.com/GAMMA-UMD/pygsound.git
cd pygsound
pip3 install .
```

---

## GPU Acceleration (OptiX)

GPU acceleration uses NVIDIA OptiX for ray tracing. This provides identical results to CPU but can be faster for complex scenes.

### Requirements
- NVIDIA GPU (RTX series recommended)
- NVIDIA Driver 525.60.13 or newer
- CUDA Toolkit 11.x or 12.x
- OptiX SDK 7.x (tested with 7.7.0)

### Step 1: Install CUDA Toolkit

```bash
# Check if CUDA is installed
nvcc --version

# If not installed, follow NVIDIA's guide:
# https://developer.nvidia.com/cuda-downloads
```

### Step 2: Download OptiX SDK

1. Go to https://developer.nvidia.com/designworks/optix/download
2. Create/login to NVIDIA Developer account
3. Download OptiX SDK 7.7.0 (or compatible version)
4. Extract to a location, e.g.:

```bash
# Linux
chmod +x NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh
./NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh --prefix=/path/to/install --skip-license

# The SDK will be extracted to something like:
# /path/to/install/NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64/
```

### Step 3: Configure CMakeLists.txt

Edit `CMakeLists.txt` to add your OptiX SDK path. Find the `find_path(OPTIX_INCLUDE_DIR_FOUND ...)` section and add your path:

```cmake
find_path(OPTIX_INCLUDE_DIR_FOUND optix.h
    HINTS ${OPTIX_INCLUDE_DIR}
    PATHS
        # Add your OptiX SDK path here:
        /path/to/NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64/include
        # Or use relative path from project root:
        ${CMAKE_SOURCE_DIR}/../NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64/include
        # Default paths:
        /usr/local/optix/include
        /usr/include
        $ENV{OPTIX_ROOT}/include
        $ENV{OptiX_INSTALL_DIR}/include
)
```

Alternatively, set environment variable before building:
```bash
export OPTIX_INCLUDE_DIR=/path/to/NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64/include
```

### Step 4: Build with GPU Support

```bash
cd pygsound
pip3 install .
```

The build system will automatically detect OptiX and enable GPU support.

### Step 5: Verify GPU Support

```python
import pygsound as ps

# Check if GPU is available
print(ps.is_gpu_available())  # True if OptiX is working
print(ps.device_info())       # Shows GPU details
```

---

## Usage

### Basic Usage (CPU)

```python
import pygsound as ps

# Create room
mesh = ps.createbox(10, 10, 10, 0.5, 0.5)  # 10x10x10 room

# Configure simulation
ctx = ps.Context()
ctx.diffuse_count = 10000
ctx.specular_count = 2000
ctx.threads_count = 4

# Create scene
scene = ps.Scene()
scene.setMesh(mesh)

# Run simulation
result = scene.getPathData(
    [[2.0, 2.0, 2.0]],  # source position
    [[8.0, 8.0, 8.0]],  # listener position
    ctx
)

print(f"Found {result['path_data'][0]['num_paths']} paths")
```

### GPU Usage

```python
import pygsound as ps

# Method 1: Per-call GPU flag
result = scene.getPathData(
    [[2.0, 2.0, 2.0]], [[8.0, 8.0, 8.0]], ctx,
    use_gpu=True  # Use GPU for this call
)

# Method 2: Device enum
result = scene.getPathData(
    [[2.0, 2.0, 2.0]], [[8.0, 8.0, 8.0]], ctx,
    use_gpu=True
)

# Check device info
print(ps.device_info())
```

### Batch Processing

Process multiple configurations efficiently:

```python
import pygsound as ps

mesh = ps.createbox(10, 10, 10, 0.5, 0.5)
ctx = ps.Context()
ctx.diffuse_count = 5000
ctx.specular_count = 1000

# Define multiple source/listener configurations
source_positions = [
    [[2.0, 2.0, 2.0]],  # Config 1
    [[3.0, 3.0, 3.0]],  # Config 2
    [[4.0, 4.0, 4.0]],  # Config 3
]

listener_positions = [
    [[8.0, 8.0, 8.0]],  # Config 1
    [[7.0, 7.0, 7.0]],  # Config 2
    [[6.0, 6.0, 6.0]],  # Config 3
]

# Batch process (CPU or GPU)
results = ps.batch_process(
    mesh,
    source_positions,
    listener_positions,
    ctx,
    device=ps.Device.GPU  # or ps.Device.CPU
)

for i, result in enumerate(results):
    print(f"Config {i+1}: {result['path_data'][0]['num_paths']} paths")
```

### Device Configuration API

```python
import pygsound as ps

# Device enum
ps.Device.CPU   # CPU ray tracing
ps.Device.GPU   # GPU (OptiX) ray tracing
ps.Device.AUTO  # Auto-select based on availability

# Check GPU availability
ps.is_gpu_available()    # Returns True/False
ps.device_info()         # Returns detailed info string

# Get effective device (resolves AUTO)
ps.get_effective_device()  # Returns Device.CPU or Device.GPU
```

---

## Important Notes

### Context Configuration
**Always set `specular_count`** - Without setting `ctx.specular_count`, propagation may hang:
```python
ctx = ps.Context()
ctx.diffuse_count = 5000
ctx.specular_count = 1000  # Required!
ctx.threads_count = 1
```

### Deterministic Results
For reproducible CPU vs GPU comparisons, create **fresh** mesh, context, and scene objects for each run:
```python
# Fresh objects ensure identical random state
mesh = ps.createbox(10, 10, 10, 0.5, 0.5)
ctx = ps.Context()
ctx.diffuse_count = 500
ctx.specular_count = 100
ctx.threads_count = 1
scene = ps.Scene()
scene.setMesh(mesh)
result = scene.getPathData([[2.0, 2.0, 2.0]], [[8.0, 8.0, 8.0]], ctx, use_gpu=True)
```

### GPU Memory
Ray tracing is **not GRAM-intensive**. A typical room uses < 1MB of GPU memory for acceleration structures.

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/rigorous_comparison_test.py -v

# Run with output
python -m pytest tests/ -v -s
```

---

## Troubleshooting

### OptiX not found during build
- Ensure OptiX SDK path is correctly set in CMakeLists.txt or via `OPTIX_INCLUDE_DIR` environment variable
- Check that the path contains `optix.h`

### `optixInit() failed with error code: 7801`
- OptiX ABI version mismatch. Your driver may be too old for the OptiX SDK version.
- Solution: Either update your NVIDIA driver or use an older OptiX SDK (7.7.0 works with most drivers)

### Simulation hangs
- Ensure `ctx.specular_count` is set
- Use `ctx.threads_count = 1` for debugging

---

## Citations

This sound propagation engine has been used for many research works. Please cite:

```
@inproceedings{schissler2011gsound,
  title={Gsound: Interactive sound propagation for games},
  author={Schissler, Carl and Manocha, Dinesh},
  booktitle={Audio Engineering Society Conference: 41st International Conference: Audio for Games},
  year={2011},
  organization={Audio Engineering Society}
}

@article{schissler2017interactive,
  title={Interactive sound propagation and rendering for large multi-source scenes},
  author={Schissler, Carl and Manocha, Dinesh},
  journal={ACM Transactions on Graphics (TOG)},
  volume={36},
  number={1},
  pages={2},
  year={2017},
  publisher={ACM}
}

@inproceedings{9052932,
  author={Z. {Tang} and L. {Chen} and B. {Wu} and D. {Yu} and D. {Manocha}},  
  booktitle={ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},  
  title={Improving Reverberant Speech Training Using Diffuse Acoustic Simulation},   
  year={2020},  
  volume={},  
  number={},  
  pages={6969-6973},
}
```

For a complete list of relevant work, see [speech related research](https://gamma.umd.edu/researchdirections/speech/main) and [sound related research](https://gamma.umd.edu/researchdirections/sound/main).

---

## License

Copyright (C) 2010-2020 Carl Schissler, University of North Carolina at Chapel Hill.
All rights reserved.

**pygsound** is the Python package that wraps **GSound**'s codebase for efficiently computing room impulse responses (RIRs) with specular and diffuse reflections. See LICENSE.txt for details.
