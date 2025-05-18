from ultralytics import YOLO
import pandas as pd

def detect_with_yolov11n(image_path: str, output_image_path: str):
    """
    使用原生 yolov11n 模型对单张图片进行检测，
    并将结果输出到 DataFrame，显示并保存带框图像。
    """
    # 加载官方发布的 yolov11n nano 模型（会自动下载权重）
    model = YOLO('yolo11n.pt')

    # 切换到评估模式（可省略，YOLO() 默认即为 eval 模式）
    model.eval()

    # 推理——返回一个 Result 对象列表
    results = model(image_path)

    # 取列表中的第 0 个结果
    result = results[0]
    boxes = result.boxes    # 检测框信息
    names = result.names    # 类别映射表

    # 构造 DataFrame：xywh + confidence + 类别 ID + 类别名称
    df = pd.DataFrame(boxes.xywh.cpu().numpy(), columns=['x', 'y', 'w', 'h'])
    df['confidence'] = boxes.conf.cpu().numpy()        # 置信度
    df['class_id']   = boxes.cls.cpu().numpy().astype(int)  # 类别 ID（整数）
    df['class_name'] = df['class_id'].map(names)            # 类别名称

    # 打印结果表
    print(df)

    # 在窗口中显示带检测框的图片
    result.show()

    # 保存带检测框的图片到指定路径
    result.save(output_image_path)

if __name__ == '__main__':
    # 测试图片路径
    image_path = 'C:/Users/12821/Desktop/PotWise/YOLO/test images/19.png'
    # 输出带框图像的保存路径
    output_path = 'C:/Users/12821/Desktop/PotWise/YOLO/output_image_by_original_yolo.jpg'

    detect_with_yolov11n(image_path, output_path)
