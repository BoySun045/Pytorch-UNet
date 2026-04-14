import cv2
import numpy as np
import matplotlib.pyplot as plt

def overlay_smart_mask(path_origin, path_metric, alpha=0.5, threshold=20):
    """
    智能叠加：去除指标图的黑色背景，只把高亮区域叠加到原图上。
    
    参数:
        threshold: 阈值 (0-255)。p2中低于这个亮度的像素会被视为"透明"，不进行叠加。
                   建议设为 10-30 之间，以此过滤掉黑色的背景。
    """
    # 1. 读取图片
    img1 = cv2.imread(path_origin)
    img2 = cv2.imread(path_metric)
    
    if img1 is None or img2 is None:
        print("Error: Images not found.")
        return

    # 2. 统一尺寸
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 3. 准备指标数据的掩膜 (Mask)
    gray_metric = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # --- 关键步骤：生成热力图 ---
    # 我们依然可以使用 JET，因为它对高值的红/黄表现很好
    # 但我们稍后会把它的蓝色背景扣掉
    norm_metric = cv2.normalize(gray_metric, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    heatmap = cv2.applyColorMap(norm_metric, cv2.COLORMAP_JET)

    # --- 关键步骤：智能抠图 ---
    # 创建一个 mask：只有当 gray_metric > threshold 时，mask 为 True (255)
    # 这样 p2 中原本黑色的区域，就会被忽略
    _, mask = cv2.threshold(gray_metric, threshold, 255, cv2.THRESH_BINARY)
    
    # 将 mask 转为 3 通道，以便和彩色图片运算
    mask_inv = cv2.bitwise_not(mask)
    
    # 4. 图像合成
    # 区域 A: 即使是高亮区域，我们也希望它有一点透明度，能看到底下的物体
    # 公式: result = (原图 * (1-alpha)) + (热力图 * alpha)
    img1_float = img1.astype(float)
    heatmap_float = heatmap.astype(float)
    blended_region = cv2.addWeighted(img1_float, 1 - alpha, heatmap_float, alpha, 0)
    
    # 区域 B: 背景区域 (mask 为 0 的地方)，直接使用原图
    # 最终组合: 
    # 在 mask 区域使用 blended_region
    # 在 mask_inv 区域使用 原图 (img1)
    
    # 转换 mask 格式以便矩阵运算 (0 或 1)
    mask_indices = (mask > 0)
    
    # 创建最终输出图，先复制原图
    final_output = img1.copy()
    
    # 只替换 mask 为白色的区域
    final_output[mask_indices] = blended_region[mask_indices].astype(np.uint8)

    # 5. 转换 RGB 用于显示
    result_rgb = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    # 6. 绘图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img1_rgb)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(result_rgb)
    plt.title("Smart Overlay (Background Removed)")
    plt.axis('off')

    plt.tight_layout()
    
    cv2.imwrite('overlay_result.png', cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR))

# --- 使用示例 ---
# 请将文件名替换为你本地的实际路径
file_p1 = '/cluster/project/cvg/students/shangwu/Pytorch-UNet/Actmap_gt_1000/image/Actmap_MH3D_00000_28_0.jpg' 
file_p2 = '/cluster/project/cvg/students/shangwu/GEN3C/outputs/dataset_all/Actmap_MH3D_00000_28_0/outputs_multi/rgb_variance_map.png'

# 运行热力图模式 (通常报告里用这个效果最好)
overlay_smart_mask(file_p1, file_p2, alpha=0.4, threshold=20)

# 如果你只想看黑白重叠，可以把 colormap_mode 改为 False
# overlay_images(file_p1, file_p2, alpha=0.5, colormap_mode=False)