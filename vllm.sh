ls /usr/local/cuda-11.8/bin/nvcc
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export VLLM_SKIP_NVCC_CHECK=1

CUDA_VISIBLE_DEVICES=1,2 python -m vllm.entrypoints.openai.api_server --dtype auto --api-key llama --gpu-memory-utilization 0.4 --tensor-parallel-size 2 --trust-remote-code --port 8005 --model /data2/qn/KGQA/models/Qwen2-VL-7B-Instruct