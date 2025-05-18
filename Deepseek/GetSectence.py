import requests
import re
import random

def generate_lover_prompt_sentence(ingredients_dict):
    """
    根据食材字典，生成一段恋人语气、温柔提示对方先吃什么的简短句子。

    参数:
        ingredients_dict (dict): 例如 {'1': '毛肚', '2': '金针菇', '3': '虾滑'}

    返回:
        str: 生成的温柔语音提示词句子
    """
    # 从字典中随机选择一个食材
    if not ingredients_dict:
        return "Sweetie, aren’t we forgetting to add something to the hotpot?"

    selected_ingredient = random.choice(list(ingredients_dict.values()))

    # 构造 prompt（让大模型作为恋人说话）
    prompt = (
        f"You are speaking as a gentle and loving partner, enjoying hotpot together with your lover. "
        f"You see the following ingredients: {selected_ingredient}. "
        f"Say something sweet, natural, and a bit playful to suggest cooking one of them now. "
        f"Speak directly as their partner — not as an outsider. "
        f"Use spoken English that is suitable for voice narration. "
        f"Do not include emojis or underscores. Keep it under 25 words."
    )



    # 请求 API
    api_url = "http://localhost:11434/api/generate"
    model_name = "deepseek-r1:7b"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(api_url, json=payload)

    if response.status_code == 200:
        result = response.json()
        clean_text = re.sub(r'<think>.*?</think>', '', result["response"], flags=re.DOTALL)
        return clean_text.strip()
    else:
        return f"请求失败: {response.status_code} {response.text}"
