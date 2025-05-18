from PIL import Image
import os

folder = r"C:\Users\12821\Desktop\PotWise\YOLO\data\images\train"

for filename in os.listdir(folder):
    path = os.path.join(folder, filename)
    try:
        with Image.open(path) as img:
            img.verify()  # 验证图像完整性
    except Exception as e:
        print(f"{filename} 无法识别或已损坏: {e}")
