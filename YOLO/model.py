#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
本模块用于加载训练好的 YOLO 模型，并提供一个接口函数 predict(image_path)
该函数接收一张图片的路径，调用 Ultralytics 模型进行检测，返回检测结果和标注后的图片，
结果以字典形式返回，其中：
  - detections: 每个检测目标的信息（类别、置信度、边框坐标），仅包含置信度 >= 0.7 的结果
  - annotated_image: 带框和标签的图片，以 JPEG 格式编码后再转换为 base64 字符串（方便后续接口传输）

后续 Telegram Bot 可直接调用 predict() 接口，实现图片检测功能。
"""

import os
import cv2
import base64
import torch
from ultralytics import YOLO

# 模型文件路径（建议使用原始字符串或正斜杠）
MODEL_PATH = r"C:\Users\12821\Desktop\PotWise\YOLO\runs\yolo11n_hotpot6\weights\best.pt"

# 全局加载模型
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型文件不存在，请检查路径: {MODEL_PATH}")
model = YOLO(MODEL_PATH)


def predict(image_path: str) -> dict:
    """
    对输入图片进行目标检测，并返回检测结果及标注后的图片。

    参数:
      image_path: 图片文件的路径

    返回:
      dict, 格式如下：
      {
         "detections": [
             {"bbox": [x1, y1, x2, y2], "confidence": 0.95, "class": 0},
             ...
         ],
         "annotated_image": "base64编码的JPEG图片字符串"
      }
      其中仅包含置信度 >= 0.7 的检测结果，且绘制的图像只显示这些检测框。
    """
    # 调用模型进行预测，传入置信度阈值 conf=0.7，模型返回的结果中已过滤掉低于该阈值的检测框
    results = model(image_path, conf=0.7)
    result = results[0]  # 本例中只处理单张图片

    # 检查图片是否可读取
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise RuntimeError("无法读取图片")
    
    # 提取检测结果，构造返回的检测信息字典
    detections = []
    boxes = result.boxes.xyxy
    confidences = result.boxes.conf
    classes = result.boxes.cls

    # 如为 torch.Tensor，则转换为 numpy 数组
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(confidences, torch.Tensor):
        confidences = confidences.cpu().numpy()
    if isinstance(classes, torch.Tensor):
        classes = classes.cpu().numpy()
    
    for bbox, conf, cls in zip(boxes, confidences, classes):
        detection = {
            "bbox": bbox.tolist(),  # [x1, y1, x2, y2]
            "confidence": float(conf),   # 检测置信度
            "class": int(cls)         # 类别索引
        }
        detections.append(detection)
    
    # 绘图部分：由于调用模型时已传入置信度阈值，所以内置绘图方法只会绘制置信度 >= 0.7 的检测框
    # result.show()  # 显示检测结果（如果环境支持弹窗）
    temp_output_path = "temp_annotated.jpg"
    result.save(temp_output_path)  # 保存带检测框的图像到临时文件

    # 读取保存的图像，并转换为 base64 编码
    annotated_img = cv2.imread(temp_output_path)
    if annotated_img is None:
        raise RuntimeError("读取保存的标注图片失败")
    os.remove(temp_output_path)  # 删除临时文件

    success, buffer = cv2.imencode('.jpg', annotated_img)
    if not success:
        raise RuntimeError("图片编码失败")
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "detections": detections,
        "annotated_image": jpg_as_text
    }


# 调试入口：固定读取指定图片，并将检测结果保存到指定路径
if __name__ == "__main__":
    import json

    # 固定的输入图片路径
    image_path = r"C:\Users\12821\Desktop\PotWise\YOLO\test images\1.png"
    if not os.path.exists(image_path):
        print(f"图片文件不存在: {image_path}")
        exit(1)
    
    try:
        result_dict = predict(image_path)
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        exit(1)
    
    # 将标注后的图片保存为指定路径
    annotated_image_data = base64.b64decode(result_dict["annotated_image"])
    output_path = r"C:\Users\12821\Desktop\PotWise\YOLO\test_output.jpg"
    with open(output_path, "wb") as f:
        f.write(annotated_image_data)
    print(f"标注后的图片已保存至: {output_path}")
    
    # 打印检测结果
    print("检测结果:")
    print(json.dumps(result_dict["detections"], indent=2))
