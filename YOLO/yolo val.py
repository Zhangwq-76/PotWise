import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = r'C:\Users\12821\Desktop\PotWise\YOLO\per_class_AP50.csv'
df = pd.read_csv(file_path)

# 检查列名并重命名为统一格式（假设第一列是类别名，第二列是AP@0.5值）
if df.columns[0].lower() != 'class':
    df.columns = ['class', 'AP50']

# 排序：按AP50从高到低
df_sorted = df.sort_values(by='AP50', ascending=False)

# 绘图
plt.figure(figsize=(12, 6))
plt.bar(df_sorted['class'], df_sorted['AP50'])
plt.xticks(rotation=45, ha='right')
plt.title('AP@0.5 per Class')
plt.xlabel('Class')
plt.ylabel('AP@0.5')
plt.tight_layout()

# 保存图像
plt.savefig('ap50_per_class.png', dpi=300)
plt.show()
