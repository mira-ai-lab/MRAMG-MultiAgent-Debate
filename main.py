import chromadb
import argparse
import json
from openai import OpenAI
import traceback
from agents import TextAgent, VisualAgent, JudgeAgent
from utils import build_prompt_from_chroma, parse_json
import argparse
import logging
import os
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
logger = None

# CLIP 相关导入
_clip_filter = None
def get_clip_filter():
    """懒加载 CLIP 过滤器"""
    global _clip_filter
    if _clip_filter is None:
        from utils.clip_image_filter import get_clip_filter as _get
        _clip_filter = _get()
    return _clip_filter

def run_single_debate(
    conflict_type,
    disputed_item, 
    defender, 
    challenger, 
    judge, 
    query, 
    context, 
    caption, 
    text_ans, 
    visual_ans, 
    all_img_paths,
    max_round,
    caption_visual_agent,
    img_paths_visual_agent,
):
    """
    单图辩论的 SOP 流程封装
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"🔍 冲突类型: {conflict_type}")
    logger.info(f"⚖️ 开启图片冲突庭审: {disputed_item}")
    logger.info(f"👉 举证方(Defender): {defender.role_name} | 质询方(Challenger): {challenger.role_name}")
    logger.info(f"{'='*50}")

    # 判断是否需要传图片（只有视觉模型看图，文本模型只看caption）
    defender_imgs = img_paths_visual_agent if defender.role_name == "Visual Agent" else None
    defender_caption = caption_visual_agent if defender.role_name == "Visual Agent" else caption
    challenger_imgs = img_paths_visual_agent if challenger.role_name == "Visual Agent" else None
    challenger_caption = caption_visual_agent if challenger.role_name == "Visual Agent" else caption
    
    debate_history = [] # 记录辩论过程中的所有交互
    for round_id in range(max_round):
        logger.info(f"🎙️ [Defense] [Round {round_id+1}] {defender.role_name} 正在陈述理由...")
    
        defense_argument = defender.defend(
            query=query, context=context, caption=defender_caption,
            text_ans=text_ans, visual_ans=visual_ans,
            disputed_item=disputed_item, challenger_role=challenger.role_name,
            img_paths=defender_imgs, conflict_type=conflict_type,
            debate_history=debate_history,
        )
        logger.info(f"[Defense] [Round {round_id + 1}] [辩词] {defense_argument.strip()}")

        defender_decision = "DEFEND"
        defender_reasoning = defense_argument

        if round_id > 0:
            if "ACCEPT" in defense_argument.upper():
                defender_decision = "ACCEPT"
            elif "REJECT" in defense_argument.upper():
                defender_decision = "REJECT"

            if defender_decision == "ACCEPT":
                logger.info(f"\n🤝 [庭审结束] 双方达成共识，[Defense] [Round {round_id + 1}] {defender.role_name} 决定接受对方提案，无需 Judge 介入。")
                
                if conflict_type == "set_conflict":
                    final_resolution = "EXCLUDE"
                elif conflict_type == "order_conflict":
                    # 如果是顺序冲突，胜利果实就是挑战方（Challenger）主张的数组
                    challenger_key = f"{challenger.role_name.lower().replace(' ', '_')}_img_order"
                    final_resolution = disputed_item.get(challenger_key)

                return {
                    "winner": challenger.role_name,
                    "resolution": final_resolution, 
                    "reasoning": defender_reasoning,
                }
        # ==========================================
        # 质询方攻击 (Critique)
        # ==========================================
        logger.info(f"\n🔍 [Critique] [Round {round_id+1}] {challenger.role_name} 正在审查并质询...")
    
        critique_response = challenger.critique(
            query=query, context=context, caption=challenger_caption,
            text_ans=text_ans, visual_ans=visual_ans,
            disputed_item=disputed_item, defender_role=defender.role_name,
            defender_argument=defense_argument,
            img_paths=challenger_imgs, conflict_type=conflict_type, 
            debate_history=debate_history,
        )
    
        # 尝试解析 JSON 结果
        try:
            # critique_data = parse_json(critique_response)
            # decision = critique_data.get("decision", "REJECT").upper()
            # reasoning = critique_data.get("reasoning", "")
            decision = "REJECT"
            if "ACCEPT" in critique_response.upper():
                decision = "ACCEPT"
            reasoning = critique_response
            
        except Exception as e:
            logger.error(f"   [解析质询结果失败]: {e}. 原始返回: {critique_response}")
            decision = "REJECT" # 只要解析失败，强制视为拒不妥协，交由Judge裁决
            reasoning = critique_response
        
        logger.info(f"   [判决] {decision}")
        logger.info(f"   [理由] {reasoning.strip()}")

        debate_history.append({
            "round": round_id + 1,
            "defender_argument": defense_argument,
            "challenger_argument": reasoning,
        })

        if decision == "ACCEPT":
            logger.info(f"\n🤝 [庭审结束] 双方达成共识，[Defense] [Round {round_id + 1}] {challenger.role_name} 决定接受对方提案，无需 Judge 介入。")
            
            # 动态决定返回结果 (Resolution)
            if conflict_type == "set_conflict":
                final_resolution = "INCLUDE"
            elif conflict_type == "order_conflict":
                # 如果是顺序冲突，胜利果实就是防守方（Defender）主张的数组
                defender_key = f"{defender.role_name.lower().replace(' ', '_')}_img_order"
                final_resolution = disputed_item.get(defender_key)
                
            return {
                "winner": defender.role_name,  # 既然挑战方接受了，赢家就是防守方
                "resolution": final_resolution, # 返回具体的动作或数组
                "reasoning": reasoning
            }
    # ==========================================
    # 法官裁决 (Judge Verdict)
    # ==========================================
    logger.info(f"\n🔨 [Judge] [Round {round_id+1}] 双方拒绝妥协，Chief Judge 介入最终裁定...")
    # Judge 是上帝视角，必须传全套图片
    judge_verdict_str = judge.judge(
        query=query, 
        # img_paths=all_img_paths, 
        context=context, caption=caption,
        defender_role=defender.role_name, challenger_role=challenger.role_name,
        defender_argument=defense_argument, challenger_argument=reasoning,
        disputed_item=disputed_item, conflict_type=conflict_type,
        debate_history=debate_history,
    )
    
    try:
        judge_data = parse_json(judge_verdict_str)
        logger.info(f"   [最终胜方] {judge_data.get('winner')}")
        logger.info(f"   [处置结果] {judge_data.get('resolution')} {disputed_item}")
        logger.info(f"   [法官判词] {judge_data.get('reasoning')}")
        return judge_data
    except Exception as e:
        logger.error(f"   [解析JudgeJSON失败]: {e}. 原始返回: {judge_verdict_str}")
        return {"winner": "UNKNOWN", "resolution": "UNKNOWN", "reasoning": judge_verdict_str}


def sanitize_filename(name):
    """清理模型名称中的特殊字符，方便用作文件名"""
    # 如果是路径，那么取最后一个目录名
    if "/" in name:
        name = name.split("/")[-1]
    return name.replace("/", "-").replace(":", "-").replace(".", "_")

def parse_args():
    parser = argparse.ArgumentParser(description="MRAMG Benchmark Testing Script")
    parser.add_argument("--doc_name", type=str, required=True, help="Document name (e.g., manual, arxiv)")
    parser.add_argument("--text_model", type=str, default="gpt-4o", help="Model for Text Agent")
    parser.add_argument("--visual_model", type=str, default="gpt-4o", help="Model for Visual Agent")
    parser.add_argument("--judge_model", type=str, default="gpt-4o", help="Model for Judge Agent")
    parser.add_argument("--input_dir", type=str, default="test", help="Directory containing the input jsonl files")
    parser.add_argument("--output_dir", type=str, default="test/outputs", help="Directory to save the generated jsonl files")
    parser.add_argument("--top_k", type=int, default=10, help="Number of chunks to retrieve from ChromaDB")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API Key")
    parser.add_argument("--base_url", type=str, default="https://api.qingyuntop.top/v1", help="OpenAI API Base URL")
    parser.add_argument("--num_workers", type=int, default=5, help="并发执行的线程数 (建议 4-10 之间)")
    parser.add_argument("--model_mode", type=str, default="api", help="Model mode (api or vllm)")
    parser.add_argument("--img_server_port", type=int, default=8009, help="Port for image server")
    parser.add_argument("--use_clip", action="store_true", default=False, help="是否使用 CLIP 筛选图像")
    parser.add_argument("--clip_top_k", type=int, default=10, help="CLIP 筛选的 topk 图像数量")
    parser.add_argument("--max_round", type=int, default=1, help="最大debate轮数")

    parser.add_argument("--version", type=str, default="v2", help="版本号，v1或v2")

    return parser.parse_args()

def process_single_question(question, question_emb, doc_name, collection, text_agent, visual_agent, judge_agent, top_k, use_clip, clip_top_k, max_round, version):
    """封装单条数据的完整处理流水线"""

    chunks = collection.query(
        query_embeddings=[question_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"] 
    )
    
    content, captions_list, all_img_paths, img_name_to_id = build_prompt_from_chroma(doc_name, chunks)
    logger.info(f"检索完成 | 命中 chunks: {len(chunks['documents'][0])} | 图片数量: {len(all_img_paths)}")
    
    caption = "\n".join(captions_list)
    
    # 使用 CLIP 筛选图像
    # TODO：
    # 筛选图片后，需同时对caption做筛选，确保与图片caption一致。  done
    # prompt中对img的说明怎么改。 done
    # 需要讨论：chunk中的被筛选出去的占位符需不需要保留。 done 暂时都保留
    # 辩论时，如果defender为text agent，要给visual agent当前img。prompt怎么改？done
    # judge 给不给图，需要讨论 done 暂时不给图
    # 生成草稿后，没用的图之后还需要传吗 done 不需要
    # 解决完集合冲突后，要让text agent和visual agent重新生成草稿。 done

    img_paths_clip_filtered = all_img_paths
    caption_clip_filtered = caption
    
    if use_clip and all_img_paths:
        try:
            clip_filter = get_clip_filter()
            img_paths_clip_filtered = clip_filter.filter_images(question, all_img_paths, clip_top_k)
            logger.info(f"CLIP 筛选完成 | 保留图片数量: {len(img_paths_clip_filtered)}")

            img_name_clip_filtered = [img.split("/")[-1].split(".")[0] for img in img_paths_clip_filtered]

            img_ids_clip_filtered = [img_name_to_id[img_name] for img_name in img_name_clip_filtered]
            
            # 打包后按 id 排序
            paired = sorted(zip(img_ids_clip_filtered, img_name_clip_filtered))

            # 再拆开
            img_ids_clip_filtered, img_name_clip_filtered = zip(*paired)

            # 如果后续还需要 list，而不是 tuple
            img_ids_clip_filtered = list(img_ids_clip_filtered)
            img_name_clip_filtered = list(img_name_clip_filtered)

            captions_list_clip_filtered = [captions_list[i-1] for i in img_ids_clip_filtered]
            
            caption_clip_filtered = "\n".join(captions_list_clip_filtered)

            
        except Exception as e:
            logger.warning(f"CLIP 筛选失败，使用原始图片列表: {e}")

    # 2. 独立生成双轨草稿
    text_agent_response = text_agent.answer(question, content, caption)
    visual_agent_response = visual_agent.answer(question, img_paths_clip_filtered, content, caption_clip_filtered)
    logger.info(f"question: {question}, 生成初始双轨草稿，text agent: {text_agent_response}, visual agent: {visual_agent_response}")
    # 3. 集合冲突检测
    set_conflicts, common_images = judge_agent.detect_set_conflict(text_agent_response, visual_agent_response)
    logger.info(f"独立生成双轨草稿后，检测到集合冲突: {set_conflicts}")

    # 全局辩论账本
    set_debate_ledger = []
    order_debate_ledger = []

    # 4. 逐个化解冲突
    for conflict in set_conflicts:
        if conflict["conflict_type"] == "set_conflict":
            info = conflict["conflict_info"]
            text_only_imgs = info.get("images_only_in_text_agent_response", [])
            visual_only_imgs = info.get("images_only_in_visual_agent_response", [])
            
            # --- 场景 A: 文本智能体要求的图 (Text_Only) ---
            for img in text_only_imgs:
                img_paths_visual_agent = img_paths_clip_filtered
                caption_visual_agent = caption_clip_filtered

                # 版本2：debate时给visual agent额外传入当前img
                if version == "v2":
                    img_id = int(re.search(r'\d+', img).group())
                    disputed_img = all_img_paths[img_id-1]
                    if disputed_img not in img_paths_clip_filtered:
                        img_paths_visual_agent = [disputed_img] + img_paths_clip_filtered
                        caption_visual_agent = captions_list[img_id-1] + "\n" + caption_clip_filtered

                res = run_single_debate(
                    max_round=max_round, # 最大debate轮数
                    conflict_type="set_conflict",
                    disputed_item={"disputed_image": img},
                    defender=text_agent,
                    challenger=visual_agent,
                    judge=judge_agent,
                    query=question, 
                    context=content, 
                    caption=caption,
                    text_ans=text_agent_response, visual_ans=visual_agent_response,
                    all_img_paths=all_img_paths, 
                    caption_visual_agent=caption_visual_agent,
                    img_paths_visual_agent=img_paths_visual_agent,
                )
                set_debate_ledger.append({
                    "conflict_type": "set_conflict",
                    "target_image": img,
                    "resolution": res
                })

            # --- 场景 B: 视觉智能体强加的图 (Visual_Only) ---
            for img in visual_only_imgs:
                # 是否也需要先判断一下visual agent用的img在不在img_paths_clip_filtered中
                img_paths_visual_agent = img_paths_clip_filtered
                caption_visual_agent = caption_clip_filtered
                res = run_single_debate(
                    max_round=max_round, # 最大debate轮数
                    conflict_type="set_conflict",
                    disputed_item={"disputed_image": img},
                    defender=visual_agent,
                    challenger=text_agent,
                    judge=judge_agent,
                    query=question, 
                    context=content, 
                    caption=caption,
                    text_ans=text_agent_response, visual_ans=visual_agent_response,
                    all_img_paths=all_img_paths, 
                    caption_visual_agent=caption_visual_agent,
                    img_paths_visual_agent=img_paths_visual_agent,
                )
                set_debate_ledger.append({
                    "conflict_type": "set_conflict",
                    "target_image": img,
                    "resolution": res
                })

    # 解决顺序冲突前，根据确认好的img set重新生成草稿
    is_add_new_img = False
    for ledger in set_debate_ledger:
        if ledger["conflict_type"] == "set_conflict":
            img = ledger["target_image"]
            if ledger["resolution"].get("resolution", "INCLUDE").upper() == "INCLUDE":
                is_add_new_img = True
                common_images.append(img)

    # 对common_images排序
    if is_add_new_img:
        common_images.sort()

        common_img_paths = []
        common_img_captions_list = []
        for img in common_images:
            img_id = int(re.search(r'\d+', img).group())
            common_img_paths.append(all_img_paths[img_id-1])
            common_img_captions_list.append(captions_list[img_id-1])
        # 重新生成草稿
        # TODO: 改下prompt，要求根据给定的集合生成草稿。模版里加一个特殊要求的占位符。 测试脚本 done
        if version != "v1":
            common_img_captions = "\n".join(common_img_captions_list)
            extra_kwargs = {"stage": "post", "confirmed_images": common_images}
            text_agent_response = text_agent.answer(question, content, common_img_captions, extra_kwargs)
            visual_agent_response = visual_agent.answer(question, common_img_paths, content, common_img_captions, extra_kwargs)
            logger.info(f"\n重新生成双轨草稿，使用确认好的img set: {common_images}, 生成的草稿: text agent {text_agent_response}, visual agent {visual_agent_response}")


    order_conflict = judge_agent.detect_order_conflict(text_agent_response, visual_agent_response)
    logger.info(f"检测到顺序冲突: {order_conflict}")

    if order_conflict:
        info = order_conflict[0]["conflict_info"]
        text_order = info.get("text_order", [])
        visual_order = info.get("visual_order", [])
        
        if not text_order or not visual_order: 
            logger.warning(f"顺序冲突检测到空顺序: Text Order: {text_order}, Visual Order: {visual_order}")
        else:
            disputed_order_str = f"Text Agent Order: {text_order} vs Visual Agent Order: {visual_order}"
            disputed_item = {
                "text_agent_img_order": text_order,
                "visual_agent_img_order": visual_order,
            }
            
            img_paths_visual_agent = img_paths_clip_filtered
            caption_visual_agent = caption_clip_filtered

            res = run_single_debate(
                max_round=max_round, # 最大debate轮数
                conflict_type="order_conflict",
                disputed_item=disputed_item, 
                defender=text_agent,
                challenger=visual_agent,
                judge=judge_agent,
                query=question, context=content, caption=caption,
                text_ans=text_agent_response, visual_ans=visual_agent_response,
                all_img_paths=all_img_paths, 
                caption_visual_agent=caption_visual_agent,
                img_paths_visual_agent=img_paths_visual_agent,
            )
            order_debate_ledger.append({
                "conflict_type": "order_conflict",
                "target_sequence": disputed_order_str,
                "resolution": res
            })

    # 合并所有冲突
    all_conflicts = set_conflicts + order_conflict
    debate_ledger = set_debate_ledger + order_debate_ledger

    # 终局排版
    # v1 固定排版
    if version == "v1":
        final_output = judge_agent.synthesize(
            query=question,
            text_draft=text_agent_response, 
            # visual_draft=visual_agent_response, # 要不要加visual draft，需要讨论
            debate_ledger=debate_ledger,
            context=content, 
            caption=caption, 
        )
    else:
        if order_conflict:
            ordered_confirmed_images = order_debate_ledger[0]['resolution'].get("resolution", [])
            if not ordered_confirmed_images:
                ordered_confirmed_images = order_debate_ledger[0]
            extra_kwargs = {"confirmed_images": ordered_confirmed_images}
            final_output = judge_agent.synthesize(
                query=question,
                text_draft=text_agent_response, 
                debate_ledger=debate_ledger,
                context=content, 
                caption=caption,
                extra_kwargs=extra_kwargs,
            )
        else:
            # 若没有order冲突，也不需要重新排版了
            final_output = text_agent_response

    return text_agent_response, visual_agent_response, all_conflicts, debate_ledger, final_output, img_name_to_id

def main():
    args = parse_args()

    # 初始化 ChromaDB
    chromadb_client = chromadb.PersistentClient(path="/data2/qn/MRAMG/chromadb")
    collection = chromadb_client.get_or_create_collection(name=f"doc_{args.doc_name}")
    
    # 初始化 OpenAI Client (复用)
    client = OpenAI(
        api_key=args.api_key, 
        base_url=args.base_url, 
        timeout=600,
    )
    
    # 实例化所有 Agents (根据传入的参数配置模型)
    text_agent = TextAgent(client, model=args.text_model, model_mode=args.model_mode, img_server_port=args.img_server_port, version=args.version)
    visual_agent = VisualAgent(client, model=args.visual_model, model_mode=args.model_mode, img_server_port=args.img_server_port, version=args.version)

    # judge agent 版本暂时固定为 v1，给全部caption，不给img
    judge_agent = JudgeAgent(client, model=args.judge_model, model_mode=args.model_mode, img_server_port=args.img_server_port, version=args.version)

    # 配置文件路径
    input_filepath = os.path.join(args.input_dir, f"{args.doc_name}_mqa.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 构造输出文件名: doc_name_T-model_V-model_J-model.jsonl
    safe_t_model = sanitize_filename(args.text_model)
    safe_v_model = sanitize_filename(args.visual_model)
    safe_j_model = sanitize_filename(args.judge_model)
    output_filename = f"{args.doc_name}_T-{safe_t_model}_V-{safe_v_model}_J-{safe_j_model}.jsonl"
    output_filepath = os.path.join(args.output_dir, output_filename)

    # ===== 初始化日志 =====
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_filename = output_filename.replace(".jsonl", ".log")
    log_filepath = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filepath, encoding="utf-8")
        ]
    )

    global logger
    logger = logging.getLogger(__name__)

    logger.info(f"日志文件: {log_filepath}")

    if not os.path.exists(input_filepath):
        logger.error(f"❌ 找不到输入文件: {input_filepath}")
        return

    logger.info(f"🚀 开始批量处理! 输入: {input_filepath} | 输出: {output_filepath}")

    # 读取并逐行处理
    processed_count = 0
    error_count = 0
    
    with open(input_filepath, 'r', encoding='utf-8') as f_in, \
         open(output_filepath, 'w', encoding='utf-8') as f_out:
        
        lines = [(idx, line) for idx, line in enumerate(f_in) if line.strip()]

        def worker(line_idx, line):
            data = json.loads(line)
            question = data.get('question', '')
            question_emb = data.get('query_emb', None)

            logger.info(f"\n=======================================================")
            logger.info(f"📝 正在处理第 {line_idx + 1} 条数据 | Question: {question}")

            if question_emb is None:
                logger.warning(f"⚠ 第 {line_idx + 1} 条数据没有 query_emb，跳过推理")

                data["text_agent_response"] = None
                data["visual_agent_response"] = None
                data["conflicts"] = []
                data["debate_ledger"] = []
                data["final_output"] = None

                return line_idx, data

            text_ans, vis_ans, conflicts, ledger, final_ans, img_name_to_id = process_single_question(
                question=question,
                question_emb=question_emb,
                doc_name=args.doc_name,
                collection=collection,
                text_agent=text_agent,
                visual_agent=visual_agent,
                judge_agent=judge_agent,
                top_k=args.top_k,
                use_clip=args.use_clip,
                clip_top_k=args.clip_top_k, 
                max_round=args.max_round,
                version=args.version,
            )

            if question_emb is not None:
                del data["query_emb"]

            data["img_name_to_id"] = img_name_to_id
            data["text_agent_response"] = text_ans
            data["visual_agent_response"] = vis_ans
            data["conflicts"] = conflicts
            data["debate_ledger"] = ledger
            data["final_output"] = final_ans

            return line_idx, data

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:

            futures = [
                executor.submit(worker, idx, line)
                for idx, line in lines
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):

                try:
                    line_idx, data = future.result()

                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    f_out.flush()

                    processed_count += 1
                    logger.info(f"✅ 第 {line_idx + 1} 条数据处理成功并保存。")

                except Exception as e:
                    error_count += 1
                    logger.error(f"❌ 数据处理失败! 错误信息: {e}")
                    logger.error(traceback.format_exc())

    logger.info(f"\n🎉 评测任务全部完成！")
    logger.info(f"📊 统计: 成功处理 {processed_count} 条，失败 {error_count} 条。")
    logger.info(f"💾 结果已保存至: {output_filepath}")

if __name__ == "__main__":
    main()