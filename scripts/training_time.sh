#!/bin/bash
# pipeline timing on CPU and on 1-4 GPUs
eval "$(conda shell.bash hook)"
conda activate photod
export TF_CPP_MIN_LOG_LEVEL="3"
export CUDA_VISIBLE_DEVICES=""
python3 training_time.py
export CUDA_VISIBLE_DEVICES="0"
python3 training_time.py
export CUDA_VISIBLE_DEVICES="0,1"
python3 training_time.py
export CUDA_VISIBLE_DEVICES="0,1,3"
python3 training_time.py
export CUDA_VISIBLE_DEVICES="0,1,2,3"
python3 training_time.py
conda deactivate
