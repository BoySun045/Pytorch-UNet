import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import os

# 1. 定义变量
t_fixed = 48
z = 7.5           # 你现在的代码中 z = 5
alpha_deg = 32.0

theta = np.linspace(0, 30, 300) 

# 2. 角度转换为弧度
deg_to_rad = np.pi / 180.0

# 变量预计算
tan_angle = (alpha_deg + theta * (t_fixed / 120)) * deg_to_rad
theta_rad = theta * deg_to_rad
const_alpha_rad = alpha_deg * deg_to_rad 

# 3. 计算 Delta x
part1 = 0.5 * (t_fixed / 24) * np.sin(theta_rad)
part2 = (z - 0.5 * (t_fixed / 24) * np.cos(theta_rad)) * np.tan(tan_angle)
part3 = z * np.tan(const_alpha_rad)

delta_x = part1 + part2 - part3

# 4. 计算 FOV Gain
denominator = z * np.tan(const_alpha_rad)
fov_gain = delta_x / denominator

# 5. 绘图 (改为单张图)
# figsize 改为 (10, 6) 看起来更协调，不再需要那么高了
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制曲线
ax.plot(theta, fov_gain, color='#d62728', linewidth=2, label='FOV Gain')

# --- 关键设置 ---
# 1. 设置标题
ax.set_title(fr'FOV Gain at frame $t = {t_fixed}, z = {z}$', fontsize=16)

# 2. 设置 Y 轴为百分比格式
ax.yaxis.set_major_formatter(PercentFormatter(1.0))

# 3. 标签与网格
ax.set_xlabel(r'$\theta$ (degrees)', fontsize=14)
ax.set_ylabel('FOV Gain', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)

# 4. 图例 (保持你之前的公式写法)
# 注意：这里 ax 替代了之前的 ax2
ax.legend([r'FOV Gain $= \frac{\Delta x}{x}$'], loc='upper left', fontsize=12)

# --- 保存逻辑 ---
save_dir = "./" 
file_name = f"fov_gain_t{t_fixed}_z{int(z)}.png" # 改了一下文件名以匹配内容

os.makedirs(save_dir, exist_ok=True)
full_path = os.path.join(save_dir, file_name)

plt.tight_layout()
plt.savefig(full_path, dpi=600, bbox_inches='tight')

print(f"FOV Gain 图已成功保存至: {full_path}")
# plt.show()