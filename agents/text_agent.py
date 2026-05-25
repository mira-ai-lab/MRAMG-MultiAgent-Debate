from openai import OpenAI
from torch import storage
from .base_agent import BaseAgent
class TextAgent(BaseAgent):
    def __init__(self, client: OpenAI, model: str, img_server_port: int, model_mode: str, version: str = "v2"):
        super().__init__(client, model, role_name="Text Agent", img_server_port=img_server_port, model_mode=model_mode, version=version)

    def answer(self, query, context, caption, extra_kwargs: dict = None):
        if extra_kwargs is None:
            extra_kwargs = {}

        stage = extra_kwargs.get("stage", "start")
        confirmed_images = extra_kwargs.get("confirmed_images", '[]')

        if stage == "post":
            prompt_template = open("prompts/text_post.txt").read()
        else:
            prompt_template = open("prompts/text.txt").read()

        formatted_prompt = prompt_template.format(
            query=query,
            context=context,
            caption=caption, 
            confirmed_images=confirmed_images
        )

        # Text Agent 不需要传 img_paths
        content = self._build_content(formatted_prompt, img_paths=None)
        
        # 初始草稿要求严谨，temperature=0
        return self._call_llm(content, temperature=0.0)
        
if __name__ == "__main__":
    client = OpenAI(
        api_key="API_KEY", 
        base_url="BASE_URL", 
    )
    agent = TextAgent(client, model="gpt-5")
    query = "如何做西红柿炒鸡蛋?"
    strategy = agent.answer(query, "", "")
    print(strategy)