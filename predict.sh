#!/bin/bash

for i in {801..850}
do
  python3 predict.py -d -m /media/boysun/Extreme\ Pro/checkpoint_RGBD_miniset/checkpoint_epoch35.pth -i /media/boysun/Extreme\ Pro/MH3D_mesh_ply_val/Actmap_MH3D_00$i/image/ -o /media/boysun/Extreme\ Pro/MH3D_mesh_ply_val/Actmap_MH3D_00$i/predict_mask_rgbd -so /media/boysun/Extreme\ Pro/MH3D_mesh_ply_val/Actmap_MH3D_00$i/predict_overlay_rgbd --bilinear -c 1
  python3 predict.py -m /media/boysun/Extreme\ Pro/checkpoint_RGB_miniset/checkpoint_epoch35.pth -i /media/boysun/Extreme\ Pro/MH3D_mesh_ply_val/Actmap_MH3D_00$i/image/ -o /media/boysun/Extreme\ Pro/MH3D_mesh_ply_val/Actmap_MH3D_00$i/predict_mask_rgb -so /media/boysun/Extreme\ Pro/MH3D_mesh_ply_val/Actmap_MH3D_00$i/predict_overlay_rgb --bilinear -c 1
done
