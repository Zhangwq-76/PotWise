from ultralytics import YOLO
import pandas as pd

# 初始化模型
model = YOLO('C:/Users/12821/Desktop/PotWise/YOLO/runs/yolo11n_hotpot6/weights/best.pt')

# 将模型设置为评估模式
model.eval()

# 进行预测（以图片路径为例）
image_path = 'C:/Users/12821/Desktop/PotWise/YOLO/test images/19.png'  # 这里填写你要识别的图片路径
results = model(image_path)

# 由于results是一个列表，我们需要获取列表中的第一个元素
result = results[0]  # 获取第一个结果

# 获取检测结果
boxes = result.boxes  # 获取检测框
names = result.names  # 获取类别名称

# 将检测框信息转换为 DataFrame
df = pd.DataFrame(boxes.xywh.cpu().numpy(), columns=['x', 'y', 'w', 'h'])

# 获取置信度（每个检测框的置信度）
df['confidence'] = boxes.conf.cpu().numpy()  # 使用boxes.conf获取置信度

df['class_id'] = boxes.cls.cpu().numpy()  # 获取类别ID
df['class_name'] = df['class_id'].apply(lambda x: names[int(x)])  # 获取类别名称

# 打印输出结果
print(df)

# 生成并显示带有检测框的图像
result.show()  # 显示图像

# 保存带检测框的图像到文件
output_image_path = 'C:/Users/12821/Desktop/PotWise/YOLO/output_image.jpg'  # 你可以自定义保存路径
result.save(output_image_path)  # 保存图像
