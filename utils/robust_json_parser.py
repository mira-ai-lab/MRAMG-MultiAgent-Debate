import json
import re
import ast
from typing import Dict, Any, Optional
def parse_json(text: str) -> Optional[Dict[str, Any]]:
    def extract_json_candidates(text):
        candidates = []

        # ```json block
        matches = re.findall(r"```json\s*(.*?)\s*```", text, re.S | re.I)
        candidates.extend(matches)

        # ``` block
        matches = re.findall(r"```\s*(.*?)\s*```", text, re.S)
        candidates.extend(matches)

        # {...}
        matches = re.findall(r"\{.*?\}", text, re.S)
        candidates.extend(matches)

        candidates.append(text)

        return candidates


    def clean_json_string(s: str):

        s = s.strip()

        # 去掉尾逗号
        s = re.sub(r",\s*([}\]])", r"\1", s)

        # 修复字符串内部换行
        result = []
        in_string = False

        for c in s:
            if c == '"':
                in_string = not in_string
                result.append(c)
            elif c == "\n" and in_string:
                result.append("\\n")
            else:
                result.append(c)

        return "".join(result)


    candidates = extract_json_candidates(text)

    for cand in candidates:

        cleaned = clean_json_string(cand)

        # 1️⃣ 标准 JSON
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # 2️⃣ Python dict fallback
        try:
            return ast.literal_eval(cleaned)
        except Exception:
            pass

    return None

def extract_decision_and_reasoning(json_str: str) -> tuple[str, str]:
    """
    从JSON字符串中提取decision和reasoning
    
    Args:
        json_str: 可能包含JSON的字符串
        
    Returns:
        (decision, reasoning) 元组，解析失败时返回默认值
    """
    try:
        # 尝试健壮解析
        critique_data = parse_json(json_str)
        
        if critique_data:
            decision = critique_data.get("decision", "REJECT").upper()
            reasoning = critique_data.get("reasoning", "")
            return decision, reasoning
        else:
            # 解析失败，返回默认值
            return "REJECT", json_str
    except Exception as e:
        # 任何异常都返回默认值
        return "REJECT", json_str

if __name__ == "__main__":
    # 测试用例
    test_cases = [
        # 标准JSON
        '{"decision": "ACCEPT", "reasoning": "This is valid JSON"}',
        # Markdown代码块
        '```json\n{"decision": "ACCEPT", "reasoning": "Markdown code block"}\n```',
        # 带空格的Markdown
        '   ```   \n{"decision": "ACCEPT", "reasoning": "Spaced code block"}\n   ```   ',
        # 单引号JSON
        "{'decision': 'ACCEPT', 'reasoning': 'Single quotes'}",
        # 无引号键
        '{decision: "ACCEPT", reasoning: "Unquoted keys"}',
        # 尾部逗号
        '{"decision": "ACCEPT", "reasoning": "Trailing comma",}',
        # 包含其他文本
        'Some text before {"decision": "ACCEPT", "reasoning": "With prefix"} some text after',
        # 空字符串
        '',
        # 无效JSON
        '{invalid json}',
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print(f"Input: {repr(test_case[:100])}{'...' if len(test_case) > 100 else ''}")
        
        result = extract_dict_from_string(test_case)
        print(f"Parsed: {result}")
        
        decision, reasoning = extract_decision_and_reasoning(test_case)
        print(f"Decision: {decision}")
        print(f"Reasoning: {reasoning[:50]}{'...' if len(reasoning) > 50 else ''}")