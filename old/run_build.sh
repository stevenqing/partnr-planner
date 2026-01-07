#!/bin/bash
set -e

# Source conda
source /home/a5l/shuqing.a5l/miniconda3/etc/profile.d/conda.sh
conda activate habitat-llm

# Set compilers
export CC=/home/a5l/shuqing.a5l/miniconda3/envs/habitat-llm/bin/aarch64-conda-linux-gnu-gcc
export CXX=/home/a5l/shuqing.a5l/miniconda3/envs/habitat-llm/bin/aarch64-conda-linux-gnu-g++

# Verify compilers
echo "Using CC=$CC"
echo "Using CXX=$CXX"
$CC --version | head -1

# Build settings
export HEADLESS=1
export WITH_BULLET=1

# Go to habitat-sim
cd /home/a5l/shuqing.a5l/partnr-planner/third_party/habitat-sim

# Clean and build
rm -rf build
echo "Starting habitat-sim build at $(date)"
pip install --no-build-isolation -v . 2>&1 | tee build_output.log

echo "Build finished at $(date)"

# Verify
python -c "import habitat_sim; print('habitat_sim version:', habitat_sim.__version__)"
