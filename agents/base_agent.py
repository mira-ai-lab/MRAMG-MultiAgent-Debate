from openai import OpenAI
import json
import yaml
import base64
from pathlib import Path
from PIL import Image
import io
import requests
from img_server import get_local_image_url
import logging
logger = logging.getLogger(__name__)
class BaseAgent:
    def __init__(
        self, 
        client: OpenAI, 
        model: str, 
        role_name: str, 
        model_mode: str = "api", 
        img_server_port: int = 8009, 
        temperature: float = 0.3, 
        version: str = "v2"
    ):
        if client is None:
            raise ValueError("OpenAI client must be provided")

        self.client = client
        self.model = model
        self.role_name = role_name
        self.model_mode = model_mode
        self.temperature = temperature
        self.img_server_port = img_server_port
        self.img_server_url = f"http://127.0.0.1:{self.img_server_port}"
        self.version = version

        # 在初始化时一次性加载 YAML 配置
        with open("prompts/conflict_templates.yaml", "r", encoding="utf-8") as f:
            self.templates_config = yaml.safe_load(f)

    def _encode_image(self, img_path: str, max_size: int = 224) -> str:
        """
        将本地图片转为 base64 data URL，并压缩图片到 max_size 最大边长
        """
        img_path = Path(img_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        suffix = img_path.suffix.lower()
        if suffix in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
            format_str = "JPEG"
        elif suffix == ".png":
            mime = "image/png"
            format_str = "PNG"
        elif suffix == ".webp":
            mime = "image/webp"
            format_str = "WEBP"
        else:
            raise ValueError(f"Unsupported image format: {suffix}")

        # 打开图片并压缩
        with Image.open(img_path) as im:
            # 保持长宽比例，最大边长不超过 max_size
            im.thumbnail((max_size, max_size))
            buffered = io.BytesIO()
            # 保存到内存缓冲区
            if format_str == "JPEG":
                im.save(buffered, format=format_str, quality=85)  # 可以调整 JPEG 压缩质量
            else:
                im.save(buffered, format=format_str)
            encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return f"data:{mime};base64,{encoded}"
    # def _encode_image(self, img_path: str) -> str:
    #     """
    #     将本地图片转为 base64 data URL
    #     """
    #     img_path = Path(img_path)

    #     if not img_path.exists():
    #         raise FileNotFoundError(f"Image not found: {img_path}")

    #     suffix = img_path.suffix.lower()
    #     if suffix in [".jpg", ".jpeg"]:
    #         mime = "image/jpeg"
    #     elif suffix == ".png":
    #         mime = "image/png"
    #     elif suffix == ".webp":
    #         mime = "image/webp"
    #     else:
    #         raise ValueError(f"Unsupported image format: {suffix}")

    #     with open(img_path, "rb") as f:
    #         encoded = base64.b64encode(f.read()).decode("utf-8")

    #     return f"data:{mime};base64,{encoded}"

    def _build_content(self, text_prompt: str, img_paths: list = None) -> list:
        """
        通用多模态 payload 构造器。
        如果有 img_paths，就自动拼接图片；如果没有，就是纯文本。
        """
        if self.model_mode == "api":
            content = [{"type": "input_text", "text": text_prompt}]
            
            if img_paths:
                for img_path in img_paths:
                    encoded_img = self._encode_image(img_path)
                    content.append({
                        "type": "input_image",
                        "image_url": encoded_img
                    })
        elif self.model_mode == "vllm":
            content = [{"type": "text", "text": text_prompt}]

            if img_paths:
                for img_path in img_paths:
                    if Path(img_path).exists():
                        # 转成临时 HTTP Server URL
                        url = get_local_image_url(
                            img_path,
                            root_dir="MRAMG-Bench/IMAGE/IMAGE/images",  # 你需要在初始化时指定图片根目录
                            server_url=self.img_server_url,      # 你需要在初始化时指定 server 地址
                        )
                        # # 检查 URL 是否可访问
                        # try:
                        #     resp = requests.head(url, timeout=5)
                        #     if resp.status_code != 200:
                        #         print(f"[Warning] Image URL not reachable: {url}")
                        #         continue
                        # except requests.RequestException:
                        #     print(f"[Warning] Image URL request failed: {url}")
                        #     continue
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": url}
                        })
                    else:
                        # 如果已经是 URL
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": img_path}
                        })
        else:
            raise ValueError(f"Unknown model_mode: {self.model_mode}")

        return content
            
    
    def _call_llm(self, content: list, temperature: float = 0.3):
        """
        统一封装 LLM 调用逻辑
        """
        if self.model_mode == "api":
            response = self.client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                temperature=temperature
            )
            logger.info(f"Token usage: {response.usage}")
            return response.output_text
        elif self.model_mode == "vllm":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                temperature=temperature, 
                max_tokens=4096, 
            )
            return response.choices[0].message.content
    
    # --- 辩论核心方法 ---
    def defend(self, query, context, caption, text_ans, visual_ans, disputed_item, challenger_role, conflict_type, debate_history, img_paths=None):
        """举证方法"""
        if self.version == "v1":
            prompt_template = open("prompts/defense.txt").read()
        else:
            prompt_template = open("prompts/v2/defense.txt").read()
        defender_name = self.role_name.lower().replace(" ", "_")
        challenger_name = challenger_role.lower().replace(" ", "_")

        # CONFLICT_TEMPLATES = {
        #     "set_conflict": {
        #         "conflict_description": """We are currently resolving a conflict regarding the image placeholder: {disputed_image}. \nAs the {defender_role}, you chose to use {disputed_image}, while the {challenger_role} did not. """,
        #         # "task_description": """Your task is to DEFEND your decision to use {disputed_image}. 
        #         # Based STRICTLY on the provided Query, Context, and Image Captions, explain why inserting this specific image adds necessary value to the final output."""
        #     },
        #     "order_conflict": {
        #         "conflict_description": """We are currently resolving a sequential conflict regarding the ordering of images.\nBoth agents agreed to include these images, but disagreed on their chronological or logical sequence: \n- Your proposed sequence ({defender_role}): {defender_img_order} \n- The {challenger_role}'s proposed sequence: {challenger_img_order}""",
        #     }
        # }
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
        # 组装内容 (如果当前是 Visual Agent，可以在这里传入争议图片的 img_paths)
        content = self._build_content(formatted_prompt, img_paths)
        return self._call_llm(content, temperature=0.3)
    
    def critique(self, query, context, caption, text_ans, visual_ans, disputed_item, defender_role, defender_argument, conflict_type, debate_history, img_paths=None):
        """质询方法"""
        if self.version == "v1":
            prompt_template = open("prompts/critique.txt").read()
        else:
            prompt_template = open("prompts/v2/critique.txt").read()
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