# Habitat-LLM Installation Progress (ARM64/aarch64)

## System Information
- **Platform**: linux-aarch64 (ARM64)
- **OS**: Linux 6.4.0 (Cray HPC system)
- **GPU**: NVIDIA GH200 120GB (Grace Hopper) x3
- **CUDA**: 12.7

## Completed Steps

### 1. Conda Environment Created
```bash
conda create -n habitat-llm python=3.9 cmake -c conda-forge -y
```
- Python 3.9.23
- CMake 4.2.1

### 2. Git Submodules Initialized
```bash
git submodule sync
git submodule update --init --recursive
```
Submodules cloned:
- `third_party/habitat-lab`
- `third_party/transformers-CFG`
- `third_party/semantic_exploration`

### 3. PyTorch Installed (ARM64 + CUDA 12.4)
```bash
conda run -n habitat-llm pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
Installed versions:
- torch 2.5.1
- torchvision 0.20.1
- torchaudio 2.5.1
- numpy 1.26.4

### 4. Habitat-sim Cloned from Source
```bash
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git third_party/habitat-sim
```

### 5. Habitat-sim Dependencies Installed
```bash
conda run -n habitat-llm pip install -r third_party/habitat-sim/requirements.txt
```

### 6. EGL/OpenGL Dependencies Installed
```bash
conda install -n habitat-llm mesa-libegl-devel-cos7-aarch64 libglvnd-glx-cos7-aarch64 \
    libglvnd-egl-cos7-aarch64 xorg-libxrandr xorg-libxinerama xorg-libxcursor \
    xorg-libxi libxrender-cos7-aarch64 -c conda-forge -y
```

## Patches Applied

### CMakeLists.txt Version Fixes
Multiple CMakeLists.txt files in habitat-sim dependencies were patched to use `cmake_minimum_required(VERSION 3.5)` instead of older versions (required for CMake 4.x compatibility):

- `src/deps/zstd/build/cmake/CMakeLists.txt`
- `src/deps/assimp/CMakeLists.txt`
- `src/deps/assimp/code/CMakeLists.txt`
- `src/deps/bullet3/CMakeLists.txt`
- `src/deps/tinyobjloader/CMakeLists.txt`
- And many others in deps/

### setup.py Modification
Added EGL include path support in `third_party/habitat-sim/setup.py`:
```python
# Add EGL include path for ARM64 systems
egl_include = os.environ.get("EGL_INCLUDE_DIR")
if egl_include:
    cmake_args += ["-DEGL_INCLUDE_DIR={}".format(egl_include)]
```

## Current Status: INSTALLATION COMPLETE ✓

### Successfully Installed Components
- **habitat-sim**: 0.3.3 (built from source with GCC 13.2)
- **habitat-lab**: 0.3.3
- **habitat-baselines**: 0.3.3
- **transformers-CFG**: 0.2.1
- **habitat-llm**: 1.0

### Build Configuration Used
```bash
# Load GCC 13.2 from HPC modules (avoids GCC 15 assimp issues)
module load gcc-native/13.2

# Set compilers
export CC=$(which gcc)
export CXX=$(which g++)

# Set EGL include path
export EGL_INCLUDE_DIR=/home/a5l/shuqing.a5l/miniconda3/envs/habitat-llm/include

# Build with HEADLESS and BULLET enabled
export HEADLESS=1
export WITH_BULLET=1
```

### Installation Date
December 27, 2025

### Optional Components
- pybullet: Not installed (build failed, optional)

## Datasets Downloaded ✓

All datasets have been downloaded and configured:

- **Task Assets**: rearrange_task_assets, hab_spot_arm, hab3-episodes, habitat_humanoids
- **OVMM Objects**: data/objects_ovmm (416 MB)
- **HSSD Scene Dataset**: data/versioned_data/hssd-hab (~10 GB)
- **PartnR Episodes**: data/versioned_data/partnr_episodes (~1 GB)

### Data Directory Structure
```
data/
├── datasets/
│   ├── hssd/
│   ├── rearrange_pick/
│   ├── replica_cad/
│   └── partnr_episodes -> ../versioned_data/partnr_episodes
├── hssd-hab -> versioned_data/hssd-hab
├── humanoids/
├── models -> versioned_data/partnr_episodes/checkpoints
├── objects/
├── objects_ovmm/
├── replica_cad -> versioned_data/replica_cad_dataset
├── robots/
└── versioned_data/
    ├── hab3-episodes/
    ├── hab_fetch/
    ├── hab_spot_arm/
    ├── habitat_humanoids/
    ├── hssd-hab/
    ├── partnr_episodes/
    ├── rearrange_dataset_v2/
    ├── rearrange_pick_dataset_v0/
    ├── replica_cad_dataset/
    └── ycb/
```

## Notes

- This is an ARM64 (aarch64) system which is not officially supported by habitat-sim
- Pre-built conda packages for habitat-sim are not available for ARM64
- Building from source requires patching CMake files for compatibility
- The NVIDIA GH200 (Grace Hopper) uses ARM-based Grace CPU

## Troubleshooting

If build fails with scipy/numpy/pandas incompatibility:
```bash
pip install scipy==1.12.0
pip install numpy==1.22.0
pip install pandas==2.0.3
pip install opencv-python==4.10.0.82
```
