from openai import OpenAI
import json
from pdb import set_trace;
PROMPT_PROPOSAL = open("prompts/proposal.txt").read()
class ProposalAgent:
    def __init__(self, client, model="gpt-5-ca"):
        self.client = client
        self.model = model
        self.prompt_template = PROMPT_PROPOSAL
    def generate_proposal(self, query):
        prompt = self.prompt_template.format(query=query)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    def parse_strategy(self, strategy):
        # 解析LLM输出的策略字符串为JSON格式,若解析失败则尝试用正则表达式提取
        try:
            return json.loads(strategy)
        except json.JSONDecodeError:
            import re
            match = re.search(r'"retrieval_order":\s*"([^"]+)"', strategy)
            if match:
                return {"retrieval_order": match.group(1)}
            return None
    
    def forward(self, query):
        strategy = self.generate_proposal(query)
        parsed_strategy = self.parse_strategy(strategy)
        if parsed_strategy:
            return parsed_strategy
        else:
            return {"retrieval_order": "joint", "fusion": "late", "confidence_source": "both"}


if __name__ == "__main__":
    client = OpenAI(
        api_key="API_KEY", 
        base_url="BASE_URL", 
    )
    agent = ProposalAgent(client, model="gpt-5")
    query = "如何做西红柿炒鸡蛋?"
    strategy = agent.forward(query)
    print(strategy)