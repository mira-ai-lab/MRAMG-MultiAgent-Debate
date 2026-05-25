import os
import sys
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import BaseAgent
from openai import OpenAI
import time

class EvalAgent(BaseAgent):
    def __init__(self, client: OpenAI, model: str):
        super().__init__(client, model, role_name="Eval Agent")

    def get_score_from_response(self, response: str) -> float:
        pattern = r"<([^>]+)>(\d+\.?\d*)</\1>"
    
        # 查找所有匹配项
        matches = re.findall(pattern, response)
        if matches:
            # 提取第一个匹配到的分数（通常只有一个评分标签）
            score_str = matches[0][1]
            try:
                # 转换为float类型
                score = float(score_str)
                return score
            except ValueError:
                print(f"警告：提取的内容'{score_str}'无法转换为浮点数")
                return 0.0
        else:
            # 未找到标签时返回 None，用于重试判断
            return None

    def eval(self, prompt_template, query, context, caption, answer, max_retries=3, retry_delay=1.0):
        """
        增加重试机制：
        - max_retries: 最大重试次数
        - retry_delay: 重试间隔（秒）
        """
        formatted_prompt = prompt_template.format(
            query=query,
            context=context,
            caption=caption,
            answer=answer,
        )

        for attempt in range(1, max_retries + 1):
            try:
                content = self._build_content(formatted_prompt, img_paths=None)
                response = self._call_llm(content, temperature=0.0)
                score = self.get_score_from_response(response)
                
                if score is not None:
                    return score
                else:
                    print(f"⚠️ 第 {attempt} 次尝试未找到评分标签，重试中...")
                    
            except Exception as e:
                print(f"⚠️ 第 {attempt} 次调用 LLM 发生异常: {e}")
            
            time.sleep(retry_delay)

        # 重试结束仍未成功
        print("❌ 未能从 LLM 响应中获取评分，返回默认值 0.0")
        return 0.0