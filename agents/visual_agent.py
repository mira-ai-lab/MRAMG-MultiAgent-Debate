from openai import OpenAI
from .base_agent import BaseAgent
class VisualAgent(BaseAgent):
    def __init__(self, client: OpenAI, model: str, img_server_port: int, model_mode: str, version: str = "v2"):
        super().__init__(client, model, role_name="Visual Agent", img_server_port=img_server_port, model_mode=model_mode, version=version)

    def answer(self, query, img_paths, context, caption, extra_kwargs: dict = None):
        """生成初始图文排版草稿 (基于视觉特征)"""
        if extra_kwargs is None:
            extra_kwargs = {}

        stage = extra_kwargs.get("stage", "start")
        confirmed_images = extra_kwargs.get("confirmed_images", [])

        if stage == "post":
            prompt_template = open("prompts/visual_post.txt").read()
        else:
            if self.version == "v1":
                prompt_template = open("prompts/visual.txt").read()
            else:
                prompt_template = open("prompts/v2/visual.txt").read()
        
            
        
        formatted_prompt = prompt_template.format(
            query=query,
            context=context,
            caption=caption, 
            confirmed_images=confirmed_images
        )
        
        # Visual Agent 初始生成时必须传入所有 img_paths
        content = self._build_content(formatted_prompt, img_paths=img_paths)
        
        return self._call_llm(content, temperature=0.0)
    
    def defend(self, query, context, caption, text_ans, visual_ans, disputed_item, challenger_role, conflict_type, debate_history, img_paths=None):
        """举证方法"""
        if self.version == "v1":
            prompt_template = open("prompts/defense.txt").read()
        else:
            prompt_template = open("prompts/v2/defense_visual.txt").read()

        defender_name = self.role_name.lower().replace(" ", "_")
        challenger_name = challenger_role.lower().replace(" ", "_")

        try:
            conflict_desc_template = self.templates_config["defense"][conflict_type]["conflict_description"]
        except KeyError:
            raise ValueError(f"Template not found for action 'defense' and conflict_type '{conflict_type}'")

        # 验证disputed_item必须为dict
        if not isinstance(disputed_item, dict):
            raise ValueError("disputed_item must be a dictionary")
        
        if conflict_type == "set_conflict" and disputed_item.get("disputed_image") is None:
            raise ValueError("disputed_item must contain 'disputed_image' key")

        if conflict_type == "order_conflict" and (disputed_item.get(f"{defender_name}_img_order") is None or disputed_item.get(f"{challenger_name}_img_order") is None):
            raise ValueError(f"disputed_item must contain '{defender_name}_img_order' and '{challenger_name}_img_order' keys")
        
        conflict_desc = conflict_desc_template.format(
            disputed_image=disputed_item.get("disputed_image"),
            defender_role=self.role_name,
            challenger_role=challenger_role, 
            defender_img_order=disputed_item.get(f"{defender_name}_img_order"),
            challenger_img_order=disputed_item.get(f"{challenger_name}_img_order"),
        )

        formatted_prompt = prompt_template.format(
            defender_role=self.role_name,
            query=query,
            context=context,
            caption=caption,
            text_agent_answer=text_ans,
            visual_agent_answer=visual_ans,
            disputed_image=disputed_item.get("disputed_image"),
            conflict_description=conflict_desc,
            debate_history=debate_history,
        )
        
        content = self._build_content(formatted_prompt, img_paths)
        return self._call_llm(content, temperature=0.3)

    def critique(self, query, context, caption, text_ans, visual_ans, disputed_item, defender_role, defender_argument, conflict_type, debate_history, img_paths=None):
        """质询方法"""
        if self.version == "v1":
            prompt_template = open("prompts/critique.txt").read()
        else:
            prompt_template = open("prompts/v2/critique_visual.txt").read()

        challenger_name = self.role_name.lower().replace(" ", "_")
        defender_name = defender_role.lower().replace(" ", "_")
        
        # 2. 根据冲突类型，获取对应的子文案
        try:
            conflict_desc_template = self.templates_config["critique"][conflict_type]["conflict_description"]
        except KeyError:
            raise ValueError(f"Template not found for action 'critique' and conflict_type '{conflict_type}'")

         # 验证disputed_item必须为dict
        if not isinstance(disputed_item, dict):
            raise ValueError("disputed_item must be a dictionary")
        
        if conflict_type == "set_conflict" and disputed_item.get("disputed_image") is None:
            raise ValueError("disputed_item must contain 'disputed_image' key")

        if conflict_type == "order_conflict" and (disputed_item.get(f"{defender_name}_img_order") is None or disputed_item.get(f"{challenger_name}_img_order") is None):
            raise ValueError(f"disputed_item must contain '{defender_name}_img_order' and '{challenger_name}_img_order' keys")

        conflict_desc = conflict_desc_template.format(
            disputed_image=disputed_item.get("disputed_image"),
            defender_role=defender_role,
            challenger_role=self.role_name, 
            defender_img_order=disputed_item.get(f"{defender_name}_img_order"),
            challenger_img_order=disputed_item.get(f"{challenger_name}_img_order"),
        )

        formatted_prompt = prompt_template.format(
            challenger_role=self.role_name,
            defender_role=defender_role,
            query=query,
            context=context,
            caption=caption,
            text_agent_answer=text_ans,
            visual_agent_answer=visual_ans,
            defender_argument=defender_argument, 
            conflict_description=conflict_desc,
            debate_history=debate_history,
        )
        # 同理，支持在质询时查看图片
        content = self._build_content(formatted_prompt, img_paths)
        return self._call_llm(content, temperature=0.3)

if __name__ == "__main__":
    client = OpenAI(
        api_key="llama", 
        base_url="http://127.0.0.1:8005/v1", 
    )
    agent = VisualAgent(client, model="/data2/qn/KGQA/models/Qwen2.5-VL-72B-Instruct", img_server_port=8009, model_mode="vllm")
    query = "图中描述的是什么内容?"
    img_paths = ["MRAMG-Bench/IMAGE/IMAGE/images/ARXIV/2403_14627v2_1.png", "MRAMG-Bench/IMAGE/IMAGE/images/ARXIV/2403_14627v2_1.png"]
    context = " "
    caption = "图"

    strategy = agent.answer(query, img_paths, context, caption)
    print(strategy)