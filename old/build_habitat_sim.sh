#!/bin/bash
# Habitat-sim build script for ARM64 with conda compilers
# Run this with: salloc -n1 -c16 --mem=64G -t 02:00:00 bash build_habitat_sim.sh

set -e

echo "=== Starting habitat-sim build on $(hostname) at $(date) ==="

# Activate conda environment
source /home/a5l/shuqing.a5l/miniconda3/etc/profile.d/conda.sh
conda activate habitat-llm

# Set compiler environment variables to use conda compilers
export CC=/home/a5l/shuqing.a5l/miniconda3/envs/habitat-llm/bin/aarch64-conda-linux-gnu-gcc
export CXX=/home/a5l/shuqing.a5l/miniconda3/envs/habitat-llm/bin/aarch64-conda-linux-gnu-g++

# Verify compilers
echo "Using CC=$CC"
echo "Using CXX=$CXX"
$CC --version
$CXX --version

# Go to habitat-sim directory
cd /home/a5l/shuqing.a5l/partnr-planner/third_party/habitat-sim

# Clean build directory if exists
rm -rf build

# Set build environment
export HEADLESS=1
export WITH_BULLET=1

# Don't set EGL_INCLUDE_DIR - let CMake find it naturally
# The conda compiler knows about its own sysroot

echo "=== Starting pip install at $(date) ==="

# Build and install with verbose output
pip install --no-build-isolation -v . 2>&1 | tee build_output.log

BUILD_STATUS=$?

echo "=== Build completed with status $BUILD_STATUS at $(date) ==="

# Verify installation
if [ $BUILD_STATUS -eq 0 ]; then
    echo "=== Verifying habitat-sim installation ==="
    python -c "import habitat_sim; print('habitat_sim version:', habitat_sim.__version__)"
fi

exit $BUILD_STATUS
