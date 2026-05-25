#!/bin/bash

# ==========================================
# MRAMG Benchmark 多智能体测试启动脚本
# ==========================================

# 1. 设置 API 信息 (替换为你自己的真实 Key)
export API_KEY="your_api_key_here"
export BASE_URL="https://api.qingyuntop.top/v1"
export CUDA_VISIBLE_DEVICES=5
# export API_KEY="llama"
# export BASE_URL="http://127.0.0.1:8005/v1"
# 2. 设置评测文档与模型配置
DOC_NAMES=("manual") 
# DOC_NAMES=("wiki" "wit" "recipe")                  # e.g., manual, arxiv
# DOC_NAMES=("web" "wiki" "wit" "arxiv" "recipe" "manual")
# TEXT_MODEL="/data2/qn/KGQA/models/Qwen2-VL-7B-Instruct"
# VISUAL_MODEL="/data2/qn/KGQA/models/Qwen2-VL-7B-Instruct"
# JUDGE_MODEL="/data2/qn/KGQA/models/Qwen2-VL-7B-Instruct"

TEXT_MODEL="gpt-4o-mini"
VISUAL_MODEL="gpt-4o-mini"
JUDGE_MODEL="gpt-4o-mini"

# 3. 设置工程路径与并发度
INPUT_DIR="MRAMG-Bench/mqa_with_emb"
OUTPUT_DIR="/data2/qn/MRAMG/outputs/gpt-4o-mini/v2/ablation/max_round_1"
TOP_K=10

CLIP_TOP_K=10
NUM_WORKERS=10
VERSION="v2"

MAX_ROUND=1
MODEL_MODE="api"
# MODEL_MODE="vllm" # 调用本地部署的模型(vllm)
IMG_SERVER_PORT=8009 # 调用本地部署的模型(vllm)需要将图片先上传到图片服务器 执行 'python img_server.py'

# 4. 遍历文档列表执行测试
for DOC_NAME in "${DOC_NAMES[@]}"; do
    # 打印启动信息
    echo "========================================================"
    echo "🚀 启动 MRAMG Benchmark 评测..."
    echo "📄 当前处理文档: ${DOC_NAME}"
    echo "🤖 模型配置 (Text/Visual/Judge): ${TEXT_MODEL} / ${VISUAL_MODEL} / ${JUDGE_MODEL}"
    echo "⚡ 并发数: ${NUM_WORKERS}"
    echo "--------------------------------------------------------"

    # 执行 Python 脚本
    python main.py \
        --doc_name "${DOC_NAME}" \
        --text_model "${TEXT_MODEL}" \
        --visual_model "${VISUAL_MODEL}" \
        --judge_model "${JUDGE_MODEL}" \
        --input_dir "${INPUT_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --top_k ${TOP_K} \
        --api_key "${API_KEY}" \
        --base_url "${BASE_URL}" \
        --num_workers ${NUM_WORKERS} \
        --model_mode "${MODEL_MODE}" \
        --img_server_port ${IMG_SERVER_PORT} \
        --clip_top_k ${CLIP_TOP_K} \
        --version "${VERSION}" \
        --max_round ${MAX_ROUND} \
        --use_clip


    echo "--------------------------------------------------------"
    echo "✅ 文档 ${DOC_NAME} 运行结束。"
    echo ""
done

echo "🎉 所有文档的答案生成均已完成！"