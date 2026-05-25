import json
import os
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def parse_args():
    parser = argparse.ArgumentParser(description="Precompute Embeddings for MQA Benchmark")
    parser.add_argument("--input_dir", type=str, default="MRAMG-Bench", help="原始数据目录")
    parser.add_argument("--output_dir", type=str, default="MRAMG-Bench/mqa_with_emb", help="注入 embedding 后的新数据目录")
    parser.add_argument("--emb_model_path", type=str, default="/data2/qn/MRAMG/models/bge-m3", help="Embedding 模型路径")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"📦 正在加载 Embedding 模型: {args.emb_model_path}")
    emb_model = SentenceTransformer(args.emb_model_path)
    print("✅ 模型加载完成！")

    # 遍历输入目录下所有的 jsonl 文件
    for filename in os.listdir(args.input_dir):
        if not filename.endswith("_mqa.jsonl"):
            continue
            
        input_filepath = os.path.join(args.input_dir, filename)
        output_filepath = os.path.join(args.output_dir, filename)
        
        print(f"\n📄 正在处理文件: {filename}")
        
        # 1. 读取所有数据
        data_list = []
        questions = []
        with open(input_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                data_list.append(item)
                questions.append(item.get("question", ""))
                
        if not data_list:
            print("文件为空，跳过。")
            continue
            
        # 2. 批量计算 Embedding (极其快速)
        print(f"🧠 正在为 {len(questions)} 个问题计算 Embedding...")
        # batch_size 可以根据你的显存大小调整
        embeddings = emb_model.encode(questions, batch_size=1, show_progress_bar=True).tolist()
        
        # 3. 注入数据并写入新文件
        print(f"💾 正在保存至: {output_filepath}")
        with open(output_filepath, 'w', encoding='utf-8') as f:
            for i, item in enumerate(data_list):
                # 将算好的 embedding 存入新的字段 "query_emb"
                item["query_emb"] = embeddings[i]
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
    print("\n🎉 所有文件的 Embedding 预处理全部完成！")

if __name__ == "__main__":
    main()