# Habitat-LLM Installation Guide (GH200 ARM64/aarch64)

## System Requirements

- **Platform**: linux-aarch64 (ARM64)
- **GPU**: NVIDIA GH200 (Grace Hopper, sm90)
- **CUDA**: 12.6

## Installation Overview

The installation consists of three phases that must be completed in order:

1. **Base Environment** - Habitat-LLM, habitat-sim, habitat-lab
2. **Triton 3.2** - Built from source with LLVM/MLIR 17
3. **vLLM 0.6.4** - With Flash Attention 2

---

## Phase 1: Base Habitat-LLM Environment

### 1.1 Create Conda Environment

```bash
conda create -n habitat-llm python=3.9 cmake -c conda-forge -y
conda activate habitat-llm
```

### 1.2 Initialize Submodules

```bash
git submodule sync
git submodule update --init --recursive
```

### 1.3 Install PyTorch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 1.4 Build Habitat-sim from Source

ARM64 systems require building from source (no pre-built conda packages available).

```bash
# Clone habitat-sim
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git third_party/habitat-sim
pip install -r third_party/habitat-sim/requirements.txt

# Install EGL/OpenGL dependencies
conda install mesa-libegl-devel-cos7-aarch64 libglvnd-glx-cos7-aarch64 \
    libglvnd-egl-cos7-aarch64 xorg-libxrandr xorg-libxinerama xorg-libxcursor \
    xorg-libxi libxrender-cos7-aarch64 -c conda-forge -y

# Build with GCC 13.2
module load gcc-native/13.2
export CC=$(which gcc)
export CXX=$(which g++)
export EGL_INCLUDE_DIR=$CONDA_PREFIX/include
export HEADLESS=1
export WITH_BULLET=1

cd third_party/habitat-sim
pip install . --no-cache-dir
```

### 1.5 Install Habitat-lab and Dependencies

```bash
pip install -e third_party/habitat-lab/habitat-lab
pip install -e third_party/habitat-lab/habitat-baselines
pip install -e third_party/transformers-CFG
pip install -r requirements.txt
pip install -e .
```

### 1.6 Download Datasets

```bash
# Download task assets
python -m habitat_sim.utils.datasets_download --uids rearrange_task_assets hab_spot_arm hab3-episodes habitat_humanoids --data-path data/ --no-replace --no-prune

# Clone additional datasets
git clone https://huggingface.co/datasets/ai-habitat/OVMM_objects data/objects_ovmm --recursive
git clone -b partnr https://huggingface.co/datasets/hssd/hssd-hab data/versioned_data/hssd-hab
git clone https://huggingface.co/datasets/ai-habitat/partnr_episodes data/versioned_data/partnr_episodes

# Pull LFS files
cd data/versioned_data/hssd-hab && git lfs pull && cd ../../..
cd data/versioned_data/partnr_episodes && git lfs pull && cd ../../..

# Create symlinks
ln -s versioned_data/hssd-hab data/hssd-hab
mkdir -p data/datasets
ln -s ../versioned_data/partnr_episodes data/datasets/partnr_episodes
ln -s versioned_data/partnr_episodes/checkpoints data/models
```

---

## Phase 2: Triton 3.2 Installation

### Prerequisites

Download offline packages to `~/.triton/offline/`:

1. **LLVM 17**: Download `llvm-project-17.0.6.src.tar.xz` from [LLVM Releases](https://github.com/llvm/llvm-project/releases/tag/llvmorg-17.0.6)

2. **Triton 3.2**: Create recursive archive:
   ```bash
   git clone --recursive https://github.com/triton-lang/triton
   cd triton && git checkout v3.2.2 && git submodule update --init --recursive
   cd .. && tar -czf triton-3.2-recursive.tar.gz triton
   mv triton-3.2-recursive.tar.gz ~/.triton/offline/
   ```

### Run Installation

```bash
sbatch old/install_triton.slurm
```

This script will:
- Build LLVM/MLIR 17 from source (takes several hours)
- Build Triton 3.2 Python package
- Install to conda environment

---

## Phase 3: vLLM 0.6.4 Installation

**Important**: Run only after Triton installation completes successfully.

```bash
sbatch old/install_vllm.slurm
```

This script will:
- Upgrade PyTorch to 2.6.0 with CUDA 12.6
- Install Flash Attention 2.7.0.post2
- Install vLLM 0.6.4.post1
- Fix PyTorch version if downgraded by vLLM

---

## Final Package Versions

| Package | Version |
|---------|---------|
| Python | 3.9 |
| PyTorch | 2.6.0 (CUDA 12.6) |
| Triton | 3.2 |
| Flash Attention | 2.7.0.post2 |
| vLLM | 0.6.4.post1 |
| habitat-sim | 0.3.3 |
| habitat-lab | 0.3.3 |
| transformers-CFG | 0.2.1 |

---

## Important Notes

### Compiler Requirements
- **GCC 13.2** for habitat-sim build
- **GCC 12.3** for Triton/vLLM build (not 13.2 or 15)

### GLIBCXX Fix
If you encounter GLIBCXX errors, set:
```bash
export LD_PRELOAD="/opt/cray/pe/gcc-native/12/lib64/libstdc++.so.6"
```

### CUDA Architecture
For GH200 (sm90), use:
```bash
export TORCH_CUDA_ARCH_LIST="9.0"  # or "90;90a"
```

### vLLM Runtime Settings
```bash
export VLLM_USE_V1=0
export VLLM_TORCH_COMPILE=0
export VLLM_USE_TRITON_FLASH_ATTN=0
```

---

## Troubleshooting

### scipy/numpy/pandas Incompatibility
```bash
pip install scipy==1.12.0
pip install numpy==1.22.0
pip install pandas==2.0.3
pip install opencv-python==4.10.0.82
```

### Library Linking Issues
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## Verification

```python
import torch
print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())

import triton
print('Triton:', triton.__version__)

import flash_attn
print('Flash Attention:', flash_attn.__version__)

import vllm
print('vLLM:', vllm.__version__)

import habitat_sim
print('Habitat-sim:', habitat_sim.__version__)
```
