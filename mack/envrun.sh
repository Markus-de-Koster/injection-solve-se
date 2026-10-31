#!/bin/bash
# Run a command in the environment of the Mack et al. code, with the CUDA 11.8 / cuDNN 8.6
# libraries, ptxas and libdevice on the search paths (Keras 2.13 compiles the Adam update
# with XLA, which needs ptxas).
#
# usage: bash mack/envrun.sh python -I mack/train.py ...
HERE="$(cd "$(dirname "$0")" && pwd)"
exec pixi run --manifest-path "$HERE/pixi.toml" bash -c '
NV=$(python -I -c "import os,nvidia;print(os.path.dirname(nvidia.__path__[0]+\"/x\"))")
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$NV/cudnn/lib:${LD_LIBRARY_PATH:-}"
export PATH="$NV/cuda_nvcc/bin:$PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$NV/cuda_nvcc"
exec "$@"' _ "$@"
