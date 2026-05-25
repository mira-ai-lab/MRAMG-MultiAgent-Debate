from utils import extract_dict_from_string
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
):
    """
    单图辩论的 SOP 流程封装
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"🔍 冲突类型: {conflict_type}")
    logger.info(f"⚖️ 开启图片冲突庭审: {disputed_item}")
    logger.info(f"👉 举证方(Defender): {defender.role_name} | 质询方(Challenger): {challenger.role_name}")
    logger.info(f"{'='*50}")

    # ==========================================
    # Round 1: 举证方防守 (Defense)
    # ==========================================
    logger.info(f"🎙️ [Round 1] {defender.role_name} 正在陈述理由...")
    # 判断是否需要传图片（只有视觉模型看图，文本模型只看caption）
    defender_imgs = all_img_paths if defender.role_name == "Visual Agent" else None
    
    defense_argument = defender.defend(
        query=query, context=context, caption=caption,
        text_ans=text_ans, visual_ans=visual_ans,
        disputed_item=disputed_item, challenger_role=challenger.role_name,
        img_paths=defender_imgs, conflict_type=conflict_type,
    )
    logger.info(f"   [辩词] {defense_argument.strip()}")

    # ==========================================
    # Round 2: 质询方攻击 (Critique)
    # ==========================================
    logger.info(f"\n🔍 [Round 2] {challenger.role_name} 正在审查并质询...")
    challenger_imgs = all_img_paths if challenger.role_name == "Visual Agent" else None
    
    critique_response = challenger.critique(
        query=query, context=context, caption=caption,
        text_ans=text_ans, visual_ans=visual_ans,
        disputed_item=disputed_item, defender_role=defender.role_name,
        defender_argument=defense_argument,
        img_paths=challenger_imgs, conflict_type=conflict_type, 
    )
    
    # 尝试解析 JSON 结果
    try:
        critique_data = extract_dict_from_string(critique_response)
        decision = critique_data.get("decision", "REJECT").upper()
        reasoning = critique_data.get("reasoning", "")
    except Exception as e:
        logger.error(f"   [解析质询JSON失败]: {e}. 原始返回: {critique_response}")
        decision = "REJECT" # 只要解析失败，强制视为拒不妥协，交由Judge裁决
        reasoning = critique_response
        
    logger.info(f"   [判决] {decision}")
    logger.info(f"   [理由] {reasoning.strip()}")

    if decision == "ACCEPT":
        logger.info(f"\n🤝 [庭审结束] 双方达成共识，{challenger.role_name} 接受了提案。无需 Judge 介入。")
        
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
    # Round 3: 法官裁决 (Judge Verdict)
    # ==========================================
    logger.info(f"\n🔨 [Round 3] 双方拒绝妥协，Chief Judge 介入最终裁定...")
    # Judge 是上帝视角，必须传全套图片
    judge_verdict_str = judge.judge(
        query=query, img_paths=all_img_paths, context=context, caption=caption,
        defender_role=defender.role_name, challenger_role=challenger.role_name,
        defender_argument=defense_argument, challenger_argument=reasoning,
        disputed_item=disputed_item
    )
    
    try:
        judge_data = extract_dict_from_string(judge_verdict_str)
        logger.info(f"   [最终胜方] {judge_data.get('winner')}")
        logger.info(f"   [处置结果] {judge_data.get('resolution')} {disputed_item}")
        logger.info(f"   [法官判词] {judge_data.get('verdict_reasoning')}")
        return judge_data
    except Exception as e:
        logger.error(f"   [解析JudgeJSON失败]: {e}. 原始返回: {judge_verdict_str}")
        return {"winner": "UNKNOWN", "resolution": "UNKNOWN", "reasoning": judge_verdict_str}