# MRAMG-MultiAgent-Debate

本仓库提供一种基于多智能体辩论机制的多模态检索增强图文交织生成方法的代码与复现说明。该方法面向复杂长文档场景下的图文交织生成任务，将传统单体多模态大模型的一次性统一生成过程解耦为文本推理智能体、视觉感知智能体和全局裁决智能体，并通过双轨草稿生成、冲突检测、多轮辩论、动态视觉证据补充和全局综合机制，对图像选择、关键图像补充、冗余图像剔除和多图顺序重排进行统一优化。


---

## 1. 代码结构

```text
MRAMG-MultiAgent-Debate/
│
├── agents/
│   ├── text_agent.py          # 文本推理智能体
│   ├── visual_agent.py        # 视觉感知智能体
│   ├── judge_agent.py         # 全局裁决智能体
│   └── proposal.py            # 草稿生成与候选结果生成
│
├── debate/
│   └── debate.py              # 多轮辩论与冲突消解模块
│
├── eval/
│   ├── evaluation.py          # 评价入口脚本
│   └── metrics.py             # 评价指标计算
│
├── prompts/
│   └── *.txt                  # 各类智能体使用的提示词模板
│
├── utils/
│   └── *.py                   # 数据处理、检索、格式转换等工具函数
│
├── main.py                    # 主运行入口
├── run.sh                     # 推理运行示例脚本
├── eval.sh                    # 评价运行示例脚本
├── requirements.txt           # Python 依赖
├── README.md                  # 项目说明文件
└── .gitignore
```

---

## 2. 复现流程总览

完整模型的复现流程如下：

```text
Step 1. 克隆代码仓库
Step 2. 配置 Python 运行环境
Step 3. 准备 MRAMG-Bench 数据集
Step 4. 配置 API Key 与模型服务地址
Step 5. 运行完整模型生成图文交织结果
Step 6. 运行评价脚本计算实验指标
Step 7. 对照论文主实验结果
```

---

## 3. Step 1：克隆代码仓库

```bash
git clone https://github.com/anonymous/MRAMG-MultiAgent-Debate.git
cd MRAMG-MultiAgent-Debate
```

如果论文中给出的仓库地址不同，请将上述链接替换为实际匿名仓库链接。

---

## 4. Step 2：配置 Python 运行环境

建议使用 `conda` 创建独立环境：

```bash
conda create -n mramg python=3.10
conda activate mramg
pip install -r requirements.txt
```

如果使用本地部署的多模态大模型，请根据自己的 GPU、CUDA 和 PyTorch 版本额外安装对应依赖。

---

## 5. Step 3：准备 MRAMG-Bench 数据集

本文实验基于 MRAMG-Bench 数据集进行。请下载 MRAMG-Bench 数据集，并将其放置在项目根目录下：

```text
./MRAMG-Bench/
```

推荐的数据目录结构如下：

```text
MRAMG-Bench/
│
├── network/
├── academic/
├── lifestyle/
└── images/
```

其中，三个实验领域分别为：

```text
network      网络数据领域
academic     学术文档领域
lifestyle    生活方式领域
```

如果数据集路径不同，请在运行命令中修改 `--dataset_path` 参数，或在 `run.sh` 中修改对应路径。

---

## 6. Step 4：配置 API Key 与模型服务地址

本项目需要调用大语言模型或多模态大语言模型 API。为避免隐私泄露，本仓库不包含任何真实 API Key。

Linux / Git Bash 下可使用：

```bash
export API_KEY="your_api_key_here"
export BASE_URL="your_base_url_here"
```


请注意：

```text
1. 不要将真实 API Key 写入代码文件；
2. 不要上传 .env、config.yaml 或包含密钥的配置文件；
3. 如果使用本地 OpenAI-compatible 服务，请将 BASE_URL 设置为对应服务地址。
```

---

## 7. Step 5：运行完整模型

主程序入口为 `main.py`。论文主实验分别在 `network`、`academic` 和 `lifestyle` 三个领域上运行。

默认超参数设置如下：

```text
文本切片长度：256
文本检索 Top-K：10
CLIP 图像预筛选 Top-N：10
最大辩论轮数：1
LLM-as-a-Judge 评价模型：GPT-4o
```

### 7.1 运行网络数据领域实验

```bash
python main.py \
    --dataset_path ./MRAMG-Bench \
    --domain network \
    --model GPT-4o \
    --top_k 10 \
    --top_n 10 \
    --debate_round 1 \
    --api_key ${API_KEY} \
    --base_url ${BASE_URL} \
    --output_dir ./outputs/network_gpt4o
```

### 7.2 运行学术文档领域实验

```bash
python main.py \
    --dataset_path ./MRAMG-Bench \
    --domain academic \
    --model GPT-4o \
    --top_k 10 \
    --top_n 10 \
    --debate_round 1 \
    --api_key ${API_KEY} \
    --base_url ${BASE_URL} \
    --output_dir ./outputs/academic_gpt4o
```

### 7.3 运行生活方式领域实验

```bash
python main.py \
    --dataset_path ./MRAMG-Bench \
    --domain lifestyle \
    --model GPT-4o \
    --top_k 10 \
    --top_n 10 \
    --debate_round 1 \
    --api_key ${API_KEY} \
    --base_url ${BASE_URL} \
    --output_dir ./outputs/lifestyle_gpt4o
```

如果需要复现 GPT-4o-mini 结果，可将命令中的：

```bash
--model GPT-4o
```

替换为：

```bash
--model GPT-4o-mini
```

也可以直接修改并运行示例脚本：

```bash
sh run.sh
```

---

## 8. Step 6：运行评价脚本

模型生成完成后，运行评价脚本计算图像选择、文本质量和 LLM-as-a-Judge 指标。

```bash
sh eval.sh
```

也可以手动执行：

```bash
python eval/evaluation.py \
    --prediction_path ./outputs/lifestyle_gpt4o/predictions.json \
    --ground_truth_path ./MRAMG-Bench/lifestyle/ground_truth.json \
    --api_key ${API_KEY} \
    --base_url ${BASE_URL}
```

评价指标包括：

```text
Image Precision        图像选择精确率
Image Recall           图像召回率
Image F1-score         图像选择综合性能
Image Ordering Score   多图顺序一致性
Rouge-L                文本结构相似度
BERT Score             文本语义相似度
Image Relevance        图像相关性
Image Effectiveness    图像有效性
Image Position         图像位置准确性
Overall Quality        整体回答质量
Average Score          平均性能
```

评价结果默认保存到：

```text
./outputs/{domain}_{model}/evaluation_results.json
```

---

## 9. Step 7：对照论文主实验结果

完成三个领域的生成与评价后，可将 `evaluation_results.json` 中的结果与论文主实验表格进行对照：

```text
表 3：网络数据领域主实验结果
表 4：学术文档领域主实验结果
表 5：生活方式领域主实验结果
```

若复现结果与论文数值存在轻微差异，通常可能由以下原因造成：

```text
1. API 模型版本更新；
2. 大语言模型生成存在随机性；
3. LLM-as-a-Judge 评价存在轻微波动；
4. 数据集路径或数据划分不一致；
5. Top-K、Top-N 或最大辩论轮数等参数设置不同。
```

---

## 10. 输出文件说明

每次实验运行后，输出目录中通常包含以下文件：

```text
predictions.json          生成的图文交织答案
debate_logs.json          多轮辩论记录
supplement_logs.json      动态补图记录
evaluation_results.json   最终评价结果
```

其中，`predictions.json` 的格式示例如下：

```json
{
  "question_id": "example_id",
  "question": "example question",
  "generated_answer": "generated interleaved answer",
  "selected_images": ["img1", "img2"],
  "image_order": ["img1", "img2"],
  "debate_records": []
}
```

---

## 11. 可复现性说明

由于部分实验依赖 API 形式调用的大语言模型或多模态大模型，实验结果可能受到模型版本更新、API 服务状态和生成随机性的影响。为了尽可能复现论文结果，建议：

```text
1. 使用论文中相同的基座模型；
2. 使用相同的数据集划分；
3. 保持 Top-K=10、Top-N=10、最大辩论轮数 T=1；
4. 使用相同的 LLM-as-a-Judge 评价模型；
5. 使用相同的 prompt 文件；
6. 若使用本地模型，固定随机种子和推理参数。
```

---