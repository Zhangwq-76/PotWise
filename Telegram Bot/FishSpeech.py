import subprocess
import os

def generate_speech_from_text(input_text: str) -> str:
    """
    用于调用 fish-speech 模型，将输入文本生成语音文件。
    
    参数:
        input_text (str): 要生成语音的文本内容。
    
    返回:
        str: 生成的语音文件路径（E:/fish-speech-main/fake.wav）。
    """
    # Python 3.10 虚拟环境路径
    python310_path = r"E:\fish-speech-main\fishenv\env\python.exe"
    
    # 模型工作目录
    base_dir = r"E:\fish-speech-main"

    # 步骤 1：生成语义特征（text2semantic）
    text2semantic_cmd = [
        python310_path,
        os.path.join(base_dir, "fish_speech/models/text2semantic/inference.py"),
        "--text", input_text,
        "--prompt-text", "全民制作人们大家好，我是练习时长两年半的个人练习生蔡徐坤。",
        "--prompt-tokens", "fake.npy",
        "--checkpoint-path", "checkpoints/fish-speech-1.5",
        "--num-samples", "2",
        "--compile"
    ]

    # 步骤 2：将语义特征解码为音频（vqgan）
    vqgan_cmd = [
        python310_path,
        os.path.join(base_dir, "tools/vqgan/inference.py"),
        "-i", os.path.join(base_dir, "temp/codes_0.npy"),
        "--checkpoint-path", "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
    ]

    # 执行 text2semantic 推理
    try:
        print(">> 开始生成语义特征...")
        subprocess.run(text2semantic_cmd, cwd=base_dir, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"text2semantic 执行失败:\n{e.stderr}") from e

    # 执行 vqgan 解码
    try:
        print(">> 开始解码语音...")
        subprocess.run(vqgan_cmd, cwd=base_dir, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"vqgan 解码失败:\n{e.stderr}") from e

    # 返回音频路径
    output_path = os.path.join(base_dir, "fake.wav")
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"未找到生成的音频文件：{output_path}")
    
    print("✅ 语音合成完成！")
    return output_path


# =======================
# ✅ 测试代码
# =======================
if __name__ == "__main__":
    test_input = "宝宝，今天要不要加点牛肉丸？"
    try:
        audio_path = generate_speech_from_text(test_input)
        print("生成的语音文件路径：", audio_path)
    except Exception as e:
        print("出错啦：", str(e))
