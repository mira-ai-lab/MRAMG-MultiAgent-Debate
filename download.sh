#!/bin/bash
# download_model.sh
# 自动激活 conda 环境并下载 HuggingFace 模型

set -e  # 出错就退出

# === 配置参数 ===
CONDA_ENV_NAME="llama_factory"                    # 需要激活的 conda 环境
HF_ENDPOINT="https://hf-mirror.com"      # HuggingFace 镜像
BASE_DIR="/data2/qn/KGQA/models"        # 下载基目录
MODEL_NAME="Qwen/Qwen2-VL-72B-Instruct"  # 模型全名

# === 激活 conda 环境 ===
echo "Activating conda environment: $CONDA_ENV_NAME"
# 注意：先初始化 conda shell
# 对 bash/zsh 都通用
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

# === 自动生成 local-dir ===
MODEL_DIR_NAME=$(basename "$MODEL_NAME")     # 取模型名最后一部分
LOCAL_DIR="${BASE_DIR}/${MODEL_DIR_NAME}"    # 拼成本地路径

mkdir -p "$LOCAL_DIR"

echo "Downloading $MODEL_NAME to $LOCAL_DIR from $HF_ENDPOINT ..."

# === 下载命令 ===
export HF_ENDPOINT="$HF_ENDPOINT"

huggingface-cli download \
    --resume-download \
    "$MODEL_NAME" \
    --local-dir "$LOCAL_DIR" \
    --local-dir-use-symlinks False

echo "Download finished: $LOCAL_DIR"