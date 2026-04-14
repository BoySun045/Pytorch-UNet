import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

# 1. 定义变量范围
t = np.linspace(0, 120, 120)        # t: 0 到 120
theta = np.linspace(0, 30, 60)     # theta: 0 到 30
T, THETA = np.meshgrid(t, theta)    # 生成网格

# 2. 角度转换为弧度 (Python numpy 三角函数通常接受弧度)
deg_to_rad = np.pi / 180.0

z = 4

tan_angle = (32 + THETA * (T / 120)) * deg_to_rad
theta_rad = THETA * deg_to_rad
const_32_rad = 32 * deg_to_rad

# 3. 计算函数值
# 为了代码可读性，拆分为几部分
part1 = 0.5 * (T / 24) * np.sin(theta_rad)
part2 = (z - 0.5 * (T / 24) * np.cos(theta_rad)) * np.tan(tan_angle)
part3 = z * np.tan(const_32_rad)

Z = part1 + part2 - part3

# 4. 绘图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(T, THETA, Z, cmap=cm.viridis, edgecolor='none', alpha=0.9)

# 设置标签
ax.set_xlabel(r'Frame $t$')
ax.set_ylabel(r'$\theta$')
ax.set_title(r'$\Delta(x)$')

# 1. 定义你想要保存的文件夹路径 (例如你的 Cluster 路径或本地路径)
save_dir = "./" 
# 或者 Windows 本地路径: r"D:\MyResearch\Plots"

# 2. 定义文件名
file_name = "function_plot_3d.png"

# 3. 自动创建文件夹 (如果文件夹不存在，这行代码会创建它，防止报错)
os.makedirs(save_dir, exist_ok=True)

# 4. 组合完整路径
full_path = os.path.join(save_dir, file_name)

# 5. 保存图片
# dpi=300: 设置高分辨率 (论文发表通常需要 300 或 600)
# bbox_inches='tight': 自动剪裁掉周围多余的白边
plt.savefig(full_path, dpi=600, bbox_inches='tight')

print(f"图片已成功保存至: {full_path}")

# 最后再展示
# plt.show()