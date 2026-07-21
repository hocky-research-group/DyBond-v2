#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Clear build cache to avoid cross-environment conflicts
rm -rf $SCRIPT_DIR/build/dybond

# Find python interpreter
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
else
    echo "Error: Python interpreter (python3 or python) not found in the active environment!"
    exit 1
fi

# Function to check python packages
check_python_dep() {
    $PYTHON_CMD -c "import $1" &>/dev/null
    return $?
}

# Verify dependencies
MISSING_DEPS=()
if ! command -v cmake &>/dev/null; then
    MISSING_DEPS+=("cmake")
fi
if ! check_python_dep "numpy"; then
    MISSING_DEPS+=("numpy")
fi
if ! check_python_dep "gsd"; then
    MISSING_DEPS+=("gsd")
fi
if ! check_python_dep "pybind11"; then
    MISSING_DEPS+=("pybind11")
fi
if ! check_python_dep "pytest"; then
    MISSING_DEPS+=("pytest")
fi

# Check for Eigen
EIGEN_FOUND=0
for dir in "$CONDA_PREFIX/include/eigen3" "/usr/include/eigen3" "/usr/local/include/eigen3"; do
    if [ -d "$dir" ]; then
        EIGEN_FOUND=1
        break
    fi
done
if [ $EIGEN_FOUND -eq 0 ]; then
    MISSING_DEPS+=("eigen")
fi

# Auto-install missing dependencies if in a conda/mamba environment
if [ ${#MISSING_DEPS[@]} -ne 0 ] || ! check_python_dep "hoomd"; then
    if [ -n "$CONDA_PREFIX" ]; then
        echo "Missing dependencies detected: ${MISSING_DEPS[@]}. Attempting automatic installation via conda/mamba..."
        if command -v mamba &>/dev/null; then
            mamba install -y -c conda-forge cmake eigen pybind11 gsd numpy pytest ninja "hoomd=*=*gpu*"
        elif command -v conda &>/dev/null; then
            conda install -y -c conda-forge cmake eigen pybind11 gsd numpy pytest ninja "hoomd=*=*gpu*"
        else
            echo "Warning: Conda prefix is set, but neither conda nor mamba is in PATH."
        fi
    else
        echo "Warning: The following dependencies are missing: ${MISSING_DEPS[@]}"
        echo "Please install them first (e.g. conda install -c conda-forge hoomd eigen cmake pybind11 gsd numpy pytest)"
    fi
fi

# Verify HOOMD and NVCC CUDA version compatibility
HOOMD_CUDA_VERSION=$($PYTHON_CMD -c "
try:
    import hoomd
    if hoomd.version.gpu_enabled:
        print(hoomd.version.gpu_api_version)
    else:
        print('CPU')
except Exception:
    print('NONE')
" 2>/dev/null)

ACTIVE_CUDA_VERSION=""
if command -v nvcc &>/dev/null; then
    ACTIVE_CUDA_VERSION=$(nvcc --version | grep -o -E 'release [0-9]+\.[0-9]+' | cut -d' ' -f2)
fi

if [ "$HOOMD_CUDA_VERSION" != "NONE" ] && [ "$HOOMD_CUDA_VERSION" != "CPU" ] && [ -n "$ACTIVE_CUDA_VERSION" ]; then
    if [ "$HOOMD_CUDA_VERSION" != "$ACTIVE_CUDA_VERSION" ]; then
        echo "======================================================================"
        echo "WARNING: CUDA version mismatch detected!"
        echo "  - HOOMD-blue is built with CUDA version: $HOOMD_CUDA_VERSION"
        echo "  - Active CUDA compiler (nvcc) version:   $ACTIVE_CUDA_VERSION"
        echo "This mismatch may cause C++ compilation failures or runtime issues."
        echo "======================================================================"
    else
        echo "CUDA versions match: HOOMD ($HOOMD_CUDA_VERSION) == nvcc ($ACTIVE_CUDA_VERSION)"
    fi
fi

# Find HOOMD cmake directory from active python interpreter
HOOMD_CMAKE_DIR=$($PYTHON_CMD -c "
import sys, pathlib
try:
    import hoomd
    path = pathlib.Path(hoomd.__file__).resolve()
    prefix = path.parent.parent.parent.parent.parent
    cmake_dir = prefix / 'lib/cmake/hoomd'
    if cmake_dir.exists():
        print(cmake_dir)
    else:
        print(path.parent.parent / 'lib/cmake/hoomd')
except ImportError:
    print('IMPORT_ERROR', file=sys.stderr)
" 2>/dev/null)

if [ -z "$HOOMD_CMAKE_DIR" ] || [[ "$HOOMD_CMAKE_DIR" == *"IMPORT_ERROR"* ]]; then
    echo "Error: HOOMD-blue is not installed in the active Python environment ($($PYTHON_CMD --version) at $(which $PYTHON_CMD))."
    exit 1
fi

echo "Found HOOMD CMake path: $HOOMD_CMAKE_DIR"
cmake -B $SCRIPT_DIR/build/dybond -S $SCRIPT_DIR \
         -DCMAKE_CXX_FLAGS=-march=native -DCMAKE_C_FLAGS=-march=native \
         -DHOOMD_DIR=$HOOMD_CMAKE_DIR
cmake --build $SCRIPT_DIR/build/dybond --parallel 4
cmake --install $SCRIPT_DIR/build/dybond
