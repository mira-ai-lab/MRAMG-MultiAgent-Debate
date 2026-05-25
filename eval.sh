#!/bin/bash

# ==========================================
# MRAMG 多模态评估启动脚本 (Objective + LLM Subjective)
# ==========================================

# ====== 新增：HuggingFace 本地模型缓存路径 ======
export CUDA_VISIBLE_DEVICES=5
export HF_HOME="/data2/qn/MRAMG/models"
export TRANSFORMERS_CACHE="/data2/qn/MRAMG/models"

# 1. 设置 API 信息 (替换为你自己的真实 Key)
API_KEY="your_api_key_here"
BASE_URL="https://api.qingyuntop.top/v1"

# 2. 路径配置
# 输入目录 (存放你之前 test_mramg.py 跑出来的结果文件 jsonl 的文件夹)
INPUT_DIR="outputs/gpt-4o-mini/v2/done"

# 本地 BERT 模型的路径 (用于计算 BERTScore)
# 请替换为你服务器上真实的绝对路径，例如 /data2/qn/MRAMG/models/bert-base-uncased 或 roberta-large
BERT_PATH="roberta-large" 

# 3. 评估参数配置
LANG="en"             # BERTScore 的语言 ("en" 或 "zh")
DEVICE="cuda"         # 计算 BERTScore 使用的设备 ("cuda" 或 "cpu")
NUM_WORKERS=10         # LLM 评估的并发线程数 (根据 API 速率限制调整)
TOP_K=10              # 检索的 top-k 数量 (与你在 benchmark 时保持一致)

# 4. 评估结果输出配置
SUMMARY_FILE="outputs/metrics.csv" # 评估结果的输出文件路径



# 打印启动信息
echo "========================================================"
echo "📊 启动 MRAMG 综合评估流程..."
echo "📂 输入目录: ${INPUT_DIR}"
echo "🧠 BERT 模型路径: ${BERT_PATH}"
echo "⚡ 并发线程数: ${NUM_WORKERS}"
echo "--------------------------------------------------------"

# 4. 执行 Python 评估脚本
python eval/evaluation.py \
    --input_dir "${INPUT_DIR}" \
    --bert_path "${BERT_PATH}" \
    --lang "${LANG}" \
    --device "${DEVICE}" \
    --api_key "${API_KEY}" \
    --base_url "${BASE_URL}" \
    --num_workers ${NUM_WORKERS} \
    --top_k ${TOP_K} \
    --summary_file_path "${SUMMARY_FILE}"

echo "--------------------------------------------------------"
echo "✅ 评估流程全部结束。"