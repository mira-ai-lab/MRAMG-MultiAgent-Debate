import os
import json
import chromadb
from tqdm import tqdm
import argparse
from utils import build_prompt_from_chroma

def fix_img_mapping(file_path, chromadb_client, doc_name, top_k):
    """
    通过 chroma 数据库还原 img_name_to_id 字段
    """
    collection = chromadb_client.get_or_create_collection(name=f"doc_{doc_name}")
    
    # 1. 加载 query_emb 参考文件（需要 emb 才能查询 chroma）
    emb_file_path = f"MRAMG-Bench/mqa_with_emb/{doc_name}_mqa.jsonl"
    id2emb = {}
    with open(emb_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            if "query_emb" in data:
                id2emb[data["id"]] = data["query_emb"]

    all_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): all_data.append(json.loads(line))

    print(f"🔧 Fixing mapping for {os.path.basename(file_path)}...")
    
    fixed_count = 0
    for item in tqdm(all_data):
        qid = item.get("id")
        query_emb = id2emb.get(qid)
        
        if query_emb:
            # 调用你的工具函数还原映射
            chunks = collection.query(
                query_embeddings=[query_emb], 
                n_results=top_k, 
                include=["documents", "metadatas", "distances"]
            )
            # 根据你的描述，第四个返回值是正确的 img_name_to_id
            _, _, _, correct_img_map = build_prompt_from_chroma(doc_name, chunks)
            
            # 更新字段
            item["img_name_to_id"] = correct_img_map
            fixed_count += 1

    # 2. 写回原文件
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"✅ Finished. Fixed {fixed_count} records in {file_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="需要修复的 jsonl 文件路径")
    parser.add_argument("--chroma_path", type=str, default="/data2/qn/MRAMG/chromadb")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    # 初始化 DB
    chromadb_client = chromadb.PersistentClient(path=args.chroma_path)
    
    # 从文件名推断 doc_name
    file_name = os.path.basename(args.file_path)
    doc_name = file_name.split("_")[0]

    fix_img_mapping(args.file_path, chromadb_client, doc_name, args.top_k)

if __name__ == "__main__":
    main()