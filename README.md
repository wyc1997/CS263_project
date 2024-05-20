1. conda create -n [envornmnemt name here] python==3.10 pip
    I personally would prefer 3.9 but then this was given in the discussion, so be it.

2. conda install nvidia/label/cuda-12.1.0::cuda-toolkit

3. export CUDA_HOME=$CONDA_PREFIX