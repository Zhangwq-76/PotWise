import requests
import re
import os
import json

def load_ingredient_dict_and_cooktime():
    """
    从本地路径读取食材字典和对应的烹饪时间。
    要求文件目录中有两个文件：ingredients.json 和 cook_times.json
    """
    base_path = r"C:\Users\12821\Desktop\PotWise\Deepseek"

    # 模拟你的食材映射和时间数据（可换成读取json/csv文件）
    ingredient_dict = {
        0: 'meat_slice',
        1: 'beef_tripe',
        2: 'cabbage',
        3: 'tofu',
        4: 'throat_cartilage',
        5: 'beancurd_sheet',
        6: 'brain',
        7: 'fish_slice',
        8: 'fresh_shrimp',
        9: 'glass_noodle',
        10: 'soybean_roll',
        11: 'wax_gourd',
        12: 'cuttlefish',
        13: 'potato_slice',
        14: 'scallop',
        15: 'spam',
        16: 'beef_ball',
        17: 'corn',
        18: 'crab_stick',
        19: 'frozen_tofu',
        20: 'mushroom',
        21: 'needle_mushroom',
        22: 'quail_egg',
        23: 'shrimp_paste',
        24: 'beef_tongue',
        25: 'blood_curd',
        26: 'crown_daisy',
        27: 'duck_intestine',
        28: 'fish_ball',
        29: 'lettuce',
        30: 'tender_beef',
        31: 'tribulus_vegetable'
    }

    cook_times = {
        0: 15, 1: 10, 2: 30, 3: 60, 4: 12, 5: 90, 6: 60, 7: 15,
        8: 30, 9: 90, 10: 15, 11: 120, 12: 30, 13: 60, 14: 60,
        15: 30, 16: 120, 17: 180, 18: 60, 19: 90, 20: 60, 21: 30,
        22: 120, 23: 45, 24: 15, 25: 60, 26: 30, 27: 10, 28: 120,
        29: 20, 30: 10, 31: 45
    }

    return ingredient_dict, cook_times


def get_hotpot_recommendation(ingredients_dict):
    """
    构造带烹饪时间的提示词，并向大模型请求建议顺序
    """

    # 加载完整字典和时间数据
    full_ingredient_dict, cook_times = load_ingredient_dict_and_cooktime()

    # ✅ 方案 1：优化提示词结构，避免大模型添加额外食材
    prompt_lines = [
        "You are a hotpot assistant. Based on the ingredients and their cooking times, generate a cooking guide.",
        "⚠️ For each ingredient, you must include both:",
        "- Cooking Time (in seconds)",
        "- Recommended Order (based on hotpot cooking principles)",
        "",
        "Follow these general hotpot cooking principles when suggesting the recommended order:",
        "1. Thin and quick-cooking ingredients (e.g. sliced meat, leafy greens) should be placed later to avoid overcooking.",
        "2. Ingredients that release flavor (e.g. mushrooms, seafood) can be added earlier to enrich the broth.",
        "3. Ingredients that absorb flavor (e.g. tofu, noodles) can follow once the soup is flavorful.",
        "4. Try to avoid flavor contamination: cook strongly flavored items separately or later.",
        "5. Always provide Cooking Time explicitly in seconds. Do not omit it.",
        "6. Do not assume the ingredient list is complete. Make decisions only based on the input list.",
        "",
        "Do not say 'no other ingredients'. Only explain based on general rules.",
        ""
    ]



    for _, name_cn in ingredients_dict.items():
        # 查找英文名与对应烹饪时间
        item_id = None
        for k, v in full_ingredient_dict.items():
            if name_cn == v or name_cn.lower() in v.lower():  # 容错匹配
                item_id = k
                break
        if item_id is not None:
            english_name = full_ingredient_dict[item_id].replace("_", " ").title()
            cook_time = cook_times[item_id]
            prompt_lines.append(f"*Ingredient: {english_name}*")
            prompt_lines.append(f"- Cooking Time: {cook_time} seconds")
            prompt_lines.append("- Recommended Order: ")  # 让模型填充
            prompt_lines.append("")  # 空行间隔
        else:
            # 未匹配到英文名的 fallback
            prompt_lines.append(f"*Ingredient: {name_cn}*")
            prompt_lines.append("- Cooking Time: Unknown")
            prompt_lines.append("- Recommended Order: ")
            prompt_lines.append("")

    # 拼接最终提示词
    prompt = "\n".join(prompt_lines)

    # 请求本地模型接口
    api_url = "http://localhost:11434/api/generate"
    model_name = "deepseek-r1:7b"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        clean_text = re.sub(r'<think>.*?</think>', '', result["response"], flags=re.DOTALL)
        return clean_text.strip()
    except requests.exceptions.RequestException as e:
        return f"Failed to call local model: {e}"
