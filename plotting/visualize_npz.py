import os
import numpy as np
import cv2
from tqdm import tqdm

# --- 配置路径 ---
# Ground Truth 路径
gt_dir = "/cluster/project/cvg/students/shangwu/Pytorch-UNet/Actmap_gt_1000/weighted_mask"
# RGBD (Prediction) 路径
rgbd_dir = "/cluster/project/cvg/students/shangwu/Pytorch-UNet/Actmap_multi_1000/weighted_mask"

img_dir = "/cluster/project/cvg/students/shangwu/Pytorch-UNet/Actmap_multi_1000/image"

# 结果保存路径
output_base = "./comparison_results"
vis_gt_dir = os.path.join(output_base, "vis_gt")     # 保存转换后的GT图片
vis_rgbd_dir = os.path.join(output_base, "vis_rgbd") # 保存转换后的RGBD图片

# 创建输出文件夹
os.makedirs(vis_gt_dir, exist_ok=True)
os.makedirs(vis_rgbd_dir, exist_ok=True)

def main():
    # 获取所有npz文件列表
    file_list = [f for f in os.listdir(rgbd_dir) if f.endswith('.npz')]
    # 按照文件名排序，保证顺序一致
    file_list.sort() 
    
    print(f"开始处理 {len(file_list)} 个文件...")
    
    for filename in tqdm(file_list):
        path_gt_file = os.path.join(gt_dir, filename)
        path_rgbd_file = os.path.join(rgbd_dir, filename)
        
        # 检查配对文件是否存在
        if not os.path.exists(path_rgbd_file):
            print(f"Warning: {filename} not found in RGBD directory. Skipping.")
            continue
            
        # 1. 加载数据
        try:
            data_gt = np.load(path_gt_file)['weights']
            data_rgbd = np.load(path_rgbd_file)['weights']
            # import pdb; pdb.set_trace()

            gt_min = data_gt.min()
            gt_max = data_gt.max()

            if gt_max == 0:
                data_gt = np.zeros_like(data_gt)
            else:

                data_gt = (data_gt - gt_min) / (gt_max - gt_min) * 255
                data_gt = data_gt.astype(np.uint8)
            
            rgbd_min = data_rgbd.min()
            rgbd_max = data_rgbd.max()

            if rgbd_max == 0:
                data_rgbd = np.zeros_like(data_rgbd)
            else:
                data_rgbd = (data_rgbd - rgbd_min) / (rgbd_max - rgbd_min) * 255
                data_rgbd = data_rgbd.astype(np.uint8)


            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

            
        # 保存图片 (将文件名后缀从 .npz 改为 .png)
        png_name = filename.replace('.npz', '.png')
        cv2.imwrite(os.path.join(vis_gt_dir, png_name), data_gt)
        cv2.imwrite(os.path.join(vis_rgbd_dir, png_name), data_rgbd)

if __name__ == "__main__":
    main()