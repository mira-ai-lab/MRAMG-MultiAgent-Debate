from openai import OpenAI
import re
from .base_agent import BaseAgent
import json
class JudgeAgent(BaseAgent):
    def __init__(self, client: OpenAI, model: str, img_server_port: int, model_mode: str, temperature: float = 0.0, version: str = "v2"):
        super().__init__(client, model, role_name="Judge Agent", img_server_port=img_server_port, model_mode=model_mode, version=version)
        self.client = client
        self.model = model
        self.temperature = temperature

    def judge(self, query, context, caption, disputed_item, defender_role, challenger_role, defender_argument, challenger_argument, conflict_type, debate_history, img_paths=None):
        """法官裁决方法"""
        if self.version == "v1":
            prompt_template = open("prompts/judge.txt").read()
        else:
            prompt_template = open("prompts/v2/judge.txt").read()

        try:
            conflict_desc_template = self.templates_config["judge"][conflict_type]["conflict_description"]
        except KeyError:
            raise ValueError(f"Template not found for action 'judge' and conflict_type '{conflict_type}'")

        defender_name = defender_role.lower().replace(" ", "_")
        challenger_name = challenger_role.lower().replace(" ", "_")

        if not isinstance(disputed_item, dict):
            raise ValueError("disputed_item must be a dictionary")
        
        if conflict_type == "set_conflict" and disputed_item.get("disputed_image") is None:
            raise ValueError("disputed_item must contain 'disputed_image' key")

        if conflict_type == "order_conflict" and (disputed_item.get(f"{defender_name}_img_order") is None or disputed_item.get(f"{challenger_name}_img_order") is None):
            raise ValueError(f"disputed_item must contain '{defender_name}_img_order' and '{challenger_name}_img_order' keys")
        
        # 安全提取变量
        d_image = disputed_item.get("disputed_image", "")
        d_order = disputed_item.get(f"{defender_name}_img_order", "[]")
        c_order = disputed_item.get(f"{challenger_name}_img_order", "[]")

        conflict_transcript = conflict_desc_template.format(
            disputed_image=d_image,
            defender_role=defender_role,
            challenger_role=challenger_role,
            defender_img_order=d_order,
            challenger_img_order=c_order,
        )


        formatted_prompt = prompt_template.format(
            query=query,
            context=context,
            caption=caption, 
            conflict_transcript=conflict_transcript,
            defender_role=defender_role,
            challenger_role=challenger_role,
            defender_argument=defender_argument,
            challenger_argument=challenger_argument, 
            debate_history=debate_history,
        )
        
        content = self._build_content(formatted_prompt, img_paths=img_paths)
        
        return self._call_llm(content, temperature=self.temperature)
    
    def synthesize(self, query, context, caption, text_draft, debate_ledger, img_paths=None, extra_kwargs: dict = None):
        if self.version == "v1":
            prompt_template = open("prompts/synthesize.txt").read()
        else:
            prompt_template = open("prompts/v2/synthesize.txt").read()

        if extra_kwargs is None:
            extra_kwargs = {}
        confirmed_images = extra_kwargs.get("confirmed_images", [])

        formatted_prompt = prompt_template.format(
            query=query,
            context=context,
            caption=caption, 
            text_draft=text_draft,
            confirmed_images=confirmed_images,
            debate_ledger=json.dumps(debate_ledger, ensure_ascii=False, indent=2)
        )
        
        content = self._build_content(formatted_prompt, img_paths=img_paths)
        
        return self._call_llm(content, temperature=0.0)
    # 检测冲突
    def detect_conflict(self, text_agent_response, visual_agent_response):
        """
        检测文本代理和视觉代理的响应是否存在冲突。
        总共包含4种类型冲突：
        1. 选图集合冲突(P0)：宏观数量与范围分歧，该不该配图？全篇配几张图？
        2. 插入位置冲突(P2): 同图不同点，选了同一张图，但挂载的知识点/逻辑锚点不同。
        3. 局部物证冲突(P2)：同点不同图，在同一个知识点上，各自提交了不同的视觉证据。
        4. 顺序与时序冲突(P1)：因果倒置，图片的整体排列顺序违背了物理常识或时间线。
        """
        conflicts = []
        # 1. 检测集合冲突 (P0)
        set_conflicts, common_images = self.detect_set_conflict(text_agent_response, visual_agent_response)
        if set_conflicts:
            conflicts.extend(set_conflicts)
            
        # 2. 检测顺序冲突 (P1)
        order_conflicts = self.detect_order_conflict(text_agent_response, visual_agent_response)
        if order_conflicts:
            conflicts.extend(order_conflicts)
            
        return conflicts

    def detect_order_conflict(self, text_agent_response, visual_agent_response):
        """
        检测纯粹的时序与顺序冲突 (Order Conflict)。
        通过对比双方共有的图片交集，判断其相对排版顺序是否一致。
        """
        list_text_images = self._extract_images(text_agent_response)
        list_visual_images = self._extract_images(visual_agent_response)

        common_images = set(list_text_images) & set(list_visual_images)

        if len(common_images) < 2:
            return []

        text_order = []
        visual_order = []

        for img in list_text_images:
            if img in common_images and img not in text_order:
                text_order.append(img)
        for img in list_visual_images:
            if img in common_images and img not in visual_order:
                visual_order.append(img)

        conflicts = []
        if text_order != visual_order:
            conflicts.append({
                "conflict_type": "order_conflict",
                "conflict_info": {
                    "text_order": text_order,
                    "visual_order": visual_order, 
                    "common_images": list(common_images)
                }
            })

        return conflicts
    
    def detect_set_conflict(self, text_agent_response, visual_agent_response):
        """
        检测文本代理和视觉代理的响应是否存在选图集合冲突(P0)：宏观数量与范围分歧，该不该配图？全篇配几张图？
        """
        # 从text_agent_response中提取所有图片占位符
        list_text_images = self._extract_images(text_agent_response)
        # 从visual_agent_response中提取所有图片占位符
        list_visual_images = self._extract_images(visual_agent_response)
        
        # 检测 C1: 选图集合冲突 (Set Conflict)
        set_text_images = set(list_text_images)
        set_visual_images = set(list_visual_images)

        text_only = list(set_text_images - set_visual_images)
        visual_only = list(set_visual_images - set_text_images)

        has_set_conflict = bool(text_only or visual_only)

        # 交集
        common_images = list(set_text_images & set_visual_images)

        conflicts = []
        if has_set_conflict:
            conflicts.append({
                "conflict_type": "set_conflict",
                "conflict_info": {
                    "images_only_in_text_agent_response": text_only,
                    "images_only_in_visual_agent_response": visual_only
                }
            })

        return conflicts, common_images
    def _extract_images(self, text):
        """
        使用正则表达式提取文本中的所有图片占位符，保持它们在文本中的原始顺序。
        支持格式如: <img1>, <img_12>, <img0> 等。
        """
        return re.findall(r'<img_?\d+>', text)