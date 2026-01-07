#!/bin/bash
#SBATCH --job-name=habitat-sim-build
#SBATCH --output=habitat_sim_build_%j.log
#SBATCH --error=habitat_sim_build_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Habitat-sim build script for ARM64 with conda compilers
set -e

echo "=== Starting habitat-sim build on $(hostname) at $(date) ==="

# Activate conda environment
source /home/a5l/shuqing.a5l/miniconda3/etc/profile.d/conda.sh
conda activate habitat-llm

# Load GCC 13.2 from modules
module load gcc-native/13.2

# Use the module GCC
export CC=$(which gcc)
export CXX=$(which g++)

# Add conda include/lib paths that CMake needs to find packages
# But use system sysroot to avoid glibc conflicts
export CMAKE_PREFIX_PATH=/home/a5l/shuqing.a5l/miniconda3/envs/habitat-llm:$CMAKE_PREFIX_PATH

# Set EGL include path
export EGL_INCLUDE_DIR=/home/a5l/shuqing.a5l/miniconda3/envs/habitat-llm/include

# Clear conda compiler flags
unset CFLAGS CXXFLAGS LDFLAGS CMAKE_ARGS

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
