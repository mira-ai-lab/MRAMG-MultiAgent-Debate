import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import re
import argparse
import numpy as np
import threading
import concurrent.futures
import traceback
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import score as bert_score_func
from openai import OpenAI
import chromadb
import csv
from datetime import datetime
from eval.eval_agent import EvalAgent
from utils import build_prompt_from_chroma

def get_image_metrics(gt_list, pred_list):
    gt_set, pred_set = set(gt_list), set(pred_list)
    tp = len(gt_set & pred_set)
    
    if not pred_set and not gt_set:
        # 空集合情况，完全匹配
        return 1.0, 1.0, 1.0
    
    precision = tp / len(pred_set) if len(pred_set) > 0 else 0.0
    recall = tp / len(gt_set) if len(gt_set) > 0 else 1.0  # gt为空时认为召回满分
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def get_image_ordering_score(gt_list, pred_list, p1=3, p2=2, p3=1, p=3):
    n, m = len(gt_list), len(pred_list)
    if n == 0 and m == 0: return 1.0
    if n == 0: return 0.0
    intersection_count = len(set(gt_list) & set(pred_list))
    recall_factor = intersection_count / n
    if recall_factor == 0: return 0.0

    dp = np.zeros((n + 1, m + 1))
    for i in range(n + 1): dp[i][0] = i * p1
    for j in range(m + 1): dp[0][j] = j * p2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if gt_list[i-1] == pred_list[j-1] else p3
            dp[i][j] = min(dp[i-1][j] + p1, dp[i][j-1] + p2, dp[i-1][j-1] + cost)
    
    dist_ab = dp[n][m]
    norm_dist = dist_ab / max(n, m)
    penalty = (1 / p) * min(norm_dist, p)
    return max(0.0, recall_factor * (1 - penalty))

import os
import csv
import numpy as np
from datetime import datetime

def print_summary(file_name, metrics_list, llm_metrics_list=None, summary_file=None):
    """统一的打印格式，并可写入 summary_file（CSV 格式），增加评估时间和数据长度"""
    if not metrics_list:
        return
    avg = np.mean(metrics_list, axis=0)
    data_len = len(metrics_list)
    eval_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 打印到终端
    obj_pct = [round(x * 100, 2) for x in avg]

    print(f"\n[{'='*40}]")
    print(f"📊 [Summary] File: {file_name} | Data Length: {data_len} | Eval Time: {eval_time}")
    print(f"  [Objective Metrics]")
    print(f"  Img Precision: {obj_pct[0]:.2f} | Recall: {obj_pct[1]:.2f} | F1: {obj_pct[2]:.2f}")
    print(f"  Img Ordering:  {obj_pct[3]:.2f}")
    print(f"  RougeLsum:     {obj_pct[4]:.2f}")
    print(f"  BERTScore F1:  {obj_pct[5]:.2f}")
    
    llm_row_data = [None] * 4
    if llm_metrics_list and len(llm_metrics_list) > 0:
        llm_avg = np.mean(llm_metrics_list, axis=0)
        comp_pct = round((llm_avg[0] / 5) * 100, 2)
        eff_pct = round((llm_avg[1] / 5) * 100, 2)
        pos_pct = round(llm_avg[2] * 100, 2)
        rel_pct = round((llm_avg[3] / 5) * 100, 2)

        llm_row_data = [comp_pct, eff_pct, pos_pct, rel_pct]

        print(f"  [LLM Subjective Metrics (Out of 5 or 10)]")
        print(f"  Answer Quality:      {comp_pct:.2f}")
        print(f"  Image Effectiveness: {eff_pct:.2f}")
        print(f"  Image Position:      {pos_pct:.2f}")
        print(f"  Image Relevance:     {rel_pct:.2f}")
    print(f"[{'='*40}]\n")
    
    # 写入 CSV 文件
    if summary_file:
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        file_exists = os.path.exists(summary_file)
        
        with open(summary_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写表头（仅第一次）
            if not file_exists:
                header = [
                    "file_name", "data_length", "eval_time",
                    "Prec.", "Rec.", "F1", 
                    "Ord.", "R.L.", "B.S.", 
                    "Comp.", "Eff.", "Pos.", "Rel."
                ]
                writer.writerow(header)
            
            # 构建写入行
            row = [file_name, data_len, eval_time] + obj_pct + llm_row_data
            writer.writerow(row)


# ==========================================
# 3. 核心流水线：分为“客观评估”和“LLM主观评估”
# ==========================================
def process_objective_metrics(file_path, args, r_scorer):
    """阶段一：计算静态客观指标"""
    file_name = os.path.basename(file_path)
    all_data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): all_data.append(json.loads(line))
    
    if not all_data: return all_data, False

    # if "bert_score_f1" in all_data[0]:
    #     print(f"⚡ File {file_name} Objective Metrics already exist. Skipping...")
    #     return all_data, True

    print(f"🛠️ Calculating Objective Metrics for {file_name}...")
    preds_text = [d.get("final_output", "") for d in all_data]
    gts_text = [d.get("ground_truth", "") for d in all_data]

    P, R, F1 = bert_score_func(
        preds_text, gts_text, 
        model_type=args.bert_path, 
        lang=args.lang, 
        device=args.device,
        verbose=False, 
    )
    
    for i, data in enumerate(tqdm(all_data, desc="Objective Eval")):
        img_map = data.get("img_name_to_id", {})
        
        gt_imgs = []
        next_id = max(img_map.values(), default=0) + 1
        for img in data.get("images_list", []):
            if str(img) not in img_map:
                img_map[str(img)] = next_id
                next_id += 1
            gt_imgs.append(f"<img{img_map[str(img)]}>")
        
        pred_imgs = re.findall(r"<img\d+>", data.get("final_output", ""))
        
        p, r, f1_img = get_image_metrics(gt_imgs, pred_imgs)
        ord_score = get_image_ordering_score(gt_imgs, pred_imgs)
        
        rouge_res = r_scorer.score(data.get("ground_truth", ""), data.get("final_output", ""))
        rl_f1 = rouge_res['rougeLsum'].fmeasure
        
        data.update({
            "image_precision": p,
            "image_recall": r,
            "image_f1": f1_img,
            "image_ordering": ord_score,
            "rougeLsum_f1": rl_f1,
            "bert_score_f1": F1[i].item()
        })

    return all_data, True

def run_single_llm_eval(item, query_emb, eval_agent, templates_dict, collection, doc_name, top_k):
    """多线程 Worker：执行单条数据的 4 个维度 LLM 评估"""
    question = item.get("question", "")
    final_output = item.get("final_output", "")
    
    if not query_emb or not final_output:
        return item # 跳过无效数据
        
    # 1. 还原上下文 (与测试时保持一致)
    chunks = collection.query(
        query_embeddings=[query_emb], n_results=top_k, 
        include=["documents", "metadatas", "distances"]
    )
    context, caption, _, _ = build_prompt_from_chroma(doc_name, chunks)

    # 2. 依次调用 4 个维度的 LLM 评估
    for metric_key, template_str in templates_dict.items():
        try:
            llm_score = eval_agent.eval(
                prompt_template=template_str,
                query=question,
                context=context,
                caption=caption,
                answer=final_output
            )
            
            item[f"eval_{metric_key}_score"] = llm_score
                
        except Exception as e:
            item[f"eval_{metric_key}_score"] = 0.0

    return item


def process_file_pipeline(file_path, args, r_scorer, eval_agent, templates_dict, chromadb_client):
    file_name = os.path.basename(file_path)
    doc_name = file_name.split("_")[0]
    collection = chromadb_client.get_or_create_collection(name=f"doc_{doc_name}")
    
    # 阶段一：计算客观指标
    all_data, has_data = process_objective_metrics(file_path, args, r_scorer)
    if not has_data: return
    
    # 取emb
    emb_file_path = f"MRAMG-Bench/mqa_with_emb/{doc_name}_mqa.jsonl"
    id2emb = {}
    with open(emb_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            qid = data["id"]              # 每条数据的 id
            emb = data.get("query_emb")   # query_emb
            if emb is not None:
                id2emb[qid] = emb
    
    all_data_with_emb = []
    for d in all_data:
        if d["id"] in id2emb:
            all_data_with_emb.append((d, id2emb[d["id"]]))

    # 检查是否已经跑过 LLM 评估 (防止中断后重复跑)
    if "eval_answer_quality_score" in all_data[0]:
        print(f"⚡ File {file_name} LLM Metrics already exist. Skipping LLM Eval...")
    else:
        print(f"🤖 Starting LLM Evaluation (4 dimensions) for {file_name}...")
        
        # 阶段二：多线程并发调用 LLM 进行主观打分
        processed_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(run_single_llm_eval, item, query_emb, eval_agent, templates_dict, collection, doc_name, args.top_k)
                for item, query_emb in all_data_with_emb
            ]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="LLM Eval"):
                processed_data.append(future.result())
                
        # 保持原顺序
        all_data = sorted(processed_data, key=lambda x: x.get("id", ""))

    # 阶段三：回写数据并提取报告
    obj_metrics, llm_metrics = [], []
    with open(file_path, 'w', encoding='utf-8') as f:
        for d in all_data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
            
            # 收集结果用于打印总结
            obj_metrics.append([
                d.get("image_precision", 0), d.get("image_recall", 0), d.get("image_f1", 0),
                d.get("image_ordering", 0), d.get("rougeLsum_f1", 0), d.get("bert_score_f1", 0)
            ])
            llm_metrics.append([
                d.get("eval_answer_quality_score", 0),
                d.get("eval_image_effectiveness_score", 0),
                d.get("eval_image_position_score", 0),
                d.get("eval_image_relevance_score", 0)
            ])

    print_summary(file_name, obj_metrics, llm_metrics, summary_file=args.summary_file_path)


def main():
    parser = argparse.ArgumentParser(description="Multimodal Answer Evaluation Script")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with .jsonl files")
    parser.add_argument("--bert_path", type=str, required=True, help="Local path to BERT model")
    parser.add_argument("--lang", type=str, default="en", help="Language for BERTScore")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API Key")
    parser.add_argument("--base_url", type=str, default="https://api.qingyuntop.top/v1", help="OpenAI API Base URL")
    parser.add_argument("--num_workers", type=int, default=5, help="Number of threads for LLM evaluation")
    parser.add_argument("--top_k", type=int, default=10, help="Top K retrieval count")
    parser.add_argument("--summary_file_path", type=str, default="outputs/metrics.csv", help="Path to save summary metrics")
    args = parser.parse_args()

    # 1. 基础资源初始化
    r_scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    eval_agent = EvalAgent(client, model="gpt-4o")
    chromadb_client = chromadb.PersistentClient(path="/data2/qn/MRAMG/chromadb")

    # 2. 读取 4 个维度的 Prompt 模板字典
    templates_dict = {
        "answer_quality": open("eval/prompts/answer_quality.txt", encoding="utf-8").read(),
        "image_effectiveness": open("eval/prompts/image_effectiveness.txt", encoding="utf-8").read(),
        "image_position": open("eval/prompts/image_position.txt", encoding="utf-8").read(),
        "image_relevance": open("eval/prompts/image_relevance.txt", encoding="utf-8").read(),
    }

    # 3. 遍历文件执行流水线
    files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.jsonl')])
    if not files:
        print(f"No .jsonl files found in {args.input_dir}")
        return

    for filename in files:
        full_path = os.path.join(args.input_dir, filename)
        process_file_pipeline(full_path, args, r_scorer, eval_agent, templates_dict, chromadb_client)

if __name__ == "__main__":
    main()