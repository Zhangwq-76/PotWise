import os
from ultralytics import YOLO

def main():
    # 1. 选择要使用的模型权重
    model = YOLO('yolo11n.pt')

    # 设置基准路径为当前工作目录
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录

    # 更新数据集配置文件路径和输出路径
    data_path = os.path.join(base_dir, "config", "data.yaml")
    output_dir = os.path.join(base_dir, "runs")

    # 2. 开始训练
    results = model.train(
        data=data_path,
        epochs=50,
        imgsz=640,
        batch=8,
        name='yolo11n_hotpot',
        project=output_dir
    )

    # 3. 输出提示信息
    print("训练结束！模型保存在：", os.path.join(output_dir, 'yolo11n_hotpot'))

if __name__ == '__main__':
    main()
