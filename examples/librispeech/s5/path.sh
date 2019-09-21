export NEURALSP_ROOT=$PWD/../../..
export TOOL=/aifs/users/tx078/tools/neural_sp_parallel/tools
export CONDA=$TOOL/miniconda
# export KALDI_ROOT=$TOOL/kaldi
export KALDI_ROOT="/aifs/users/tx078/tools/kaldi"

# Kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$NEURALSP_ROOT/utils:$PWD/utils/:$KALDI_ROOT/tools/sctk/bin/:$TOOL/sentencepiece/build/src:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

### Python
#source $CONDA/etc/profile.d/conda.sh && conda deactivate && conda activate
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export PYTHONPATH=${PYTHONPATH}:/aifs/users/tx078/tools/neural_sp_parallel
export PYTHONIOENCODING=UTF-8
### CUDA
CUDAROOT=/aifs/users/tx078/softs/cuda
export LD_LIBRARY_PATH=$CUDAROOT/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$CUDAROOT
export CUDA_PATH=$CUDAROOT
