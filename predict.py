import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset, CarvanaDataset
from unet import UNet, UnetResnet, TwoHeadUnet
from utils.utils import plot_img_and_mask
import cv2
import albumentations as A
from utils.pred_to_line import df_to_linemap, df_wf_to_linemap, df_clsmask_to_linemap
import matplotlib.pyplot as plt

def predict_img(net,
                full_img,
                depth_img,
                device,
                scale_factor = 0.5,
                out_threshold=0.5,
                use_depth=True,
                img_size = [224,224]):
    

    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess( full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    depth = torch.from_numpy(BasicDataset.preprocess(depth_img, scale_factor, is_mask=False, is_depth=True))
    depth = depth.unsqueeze(0)
    depth = depth.to(device=device, dtype=torch.float32)

    
    # do a center crop to both image
    center_crop = transforms.CenterCrop((img_size[0], img_size[1]))
    img = center_crop(img)
    depth = center_crop(depth)

    imgs = torch.cat((img, depth), dim=1) if use_depth else img

    with torch.no_grad():
        result = net(imgs)        

    return result

def get_bin_edges(n_classes):

    if n_classes == 11:
        bin_edges = [np.log1p(1), np.log1p(50), np.log1p(150), 
                    np.log1p(300), np.log1p(450), np.log1p(750), np.log1p(1000), 
                    np.log1p(1500), np.log1p(2000),  np.log1p(2500), np.log1p(3500)]

    elif n_classes == 7:
        bin_edges = [np.log1p(1), np.log1p(100), np.log1p(300), np.log1p(500),
                np.log1p(1000), np.log1p(2000), np.log1p(3500)]

    else:
        raise ValueError("Invalid number of classes")

    return bin_edges

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', help='Filenames of input images', required=True, type=str)
    parser.add_argument('--output', '-o', metavar='OUTPUT', help='Filenames of output images')
    parser.add_argument('--save_overlay_output', '-so', metavar='SAVE_OVERLAY', help='Filenames of overlay images')
    parser.add_argument('--scale', '-s', type=float, default=0.5)
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--use_depth', '-d', action='store_true', help='Use depth images')
    parser.add_argument('--use_mono_depth', '-md', action='store_true', help='Use monodepth images')
    parser.add_argument('--head_mode', '-hm', type=str, default='segmentation', help='both or segmentation or regression')
    parser.add_argument('--regression_downsample_factor','-rdf', type=float, default=1.0, help='Downsample factor for regression head')

    return parser.parse_args()


def create_pred_vis_overlay(line_map, img, scale):

    # Normalize the line_map to [0, 255] for heatmap
    line_map_normalized = cv2.normalize(line_map, None, 0, 255, cv2.NORM_MINMAX)
    line_map_normalized = np.uint8(line_map_normalized)
    
    # Generate heatmap from normalized line_map
    heatmap = cv2.applyColorMap(line_map_normalized, cv2.COLORMAP_HOT)

    width, height = line_map_normalized.shape

    # Resize and center crop the image
    print("img size: ", img.size)
    vis_img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
    print("vis_img size: ", vis_img.size)
    center_crop = transforms.CenterCrop((width, height))
    vis_img = center_crop(vis_img)
    vis_img = np.array(vis_img)

    # Convert BGR to RGB
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on the original image
    overlay_img = cv2.addWeighted(vis_img, 0.4, heatmap, 0.6, 0)
    overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)

    # Use matplotlib to create the colorbar and combine it with the image
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display the overlay image
    ax.imshow(overlay_img)
    ax.axis('off')  # Hide axes

        # Create a colorbar with the corresponding min/max values
    cax = fig.add_axes([0.92, 0.25, 0.03, 0.5])  # Position: [left, bottom, width, height]
    norm = plt.Normalize(vmin=np.min(line_map), vmax=np.max(line_map))
    sm = plt.cm.ScalarMappable(cmap='hot', norm=norm)
    sm.set_array([])

    # Add the colorbar to the image
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Value', rotation=270, labelpad=15)

    # Add min label to the colorbar
    cbar.ax.text(1.1, 0, f'{np.min(line_map):.2f}', ha='left', va='bottom', transform=cbar.ax.transAxes)

    # convert the plt to a cv2 image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    plt.close()

    return img


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    out_files = args.output
    out_overlay_files = args.save_overlay_output
    head_mode = args.head_mode

    if args.use_depth:
        # depth file is in the same directory as the image file but with a different name, depth
        depth_files = in_files.replace('image', 'depth') if not args.use_mono_depth else in_files.replace('image', 'mono')

    if not os.path.exists(out_files):
        os.makedirs(out_files)
    if not os.path.exists(out_overlay_files):
        os.makedirs(out_overlay_files)

    if args.use_depth:  
        print("Using depth images")
        net = TwoHeadUnet(classes=args.classes,
                            in_channels=4,
                            head_config = head_mode,
                            regression_downsample_factor=args.regression_downsample_factor)
        
    else:
        net = TwoHeadUnet(classes=args.classes,
                            in_channels=3,
                            head_config = head_mode,
                            regression_downsample_factor=args.regression_downsample_factor)
        


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')
    print("model num classes: ", net.n_classes)

    # get all images in in_files folder with .jpg extension, into a list
    print(in_files)
    files = os.listdir(in_files)
    in_imgs = []
    out_imgs = []
    out_overlay_imgs = []

    for f in files:
        if f.endswith('.jpg') or f.endswith('.png'):
            in_imgs.append(os.path.join(in_files, f))
            out_imgs.append(os.path.join(out_files, f))
            out_overlay_imgs.append(os.path.join(out_overlay_files, f))

    print(in_imgs)

    

    for i, filename in enumerate(in_imgs):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)
        if args.use_depth:
            depth_filename = os.path.join(depth_files, os.path.basename(filename).replace('img', 'depth'))
            # depth is with png extension, not jpg, covert depth_filename to png
            depth_filename = depth_filename.replace('.jpg', '.png')
            depth = cv2.imread(depth_filename, cv2.IMREAD_GRAYSCALE)
            
        result = predict_img(net=net,
                           full_img=img,
                           depth_img=depth,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device,
                           use_depth=args.use_depth,
                           img_size=[224,224])
        
        if head_mode == 'df':
            # prediction is a distance field
            df_pred = result[0].squeeze(0).squeeze(0).cpu().numpy()
            line_map = df_to_linemap(df_pred, threshold=0.45)
            # save the distance field as a heatmap image
            out_filename = out_imgs[i]
            # normalize the distance field to 0-255
            df_pred = (df_pred - df_pred.min()) / (df_pred.max() - df_pred.min()) * 255
            # convert the distance field to a heatmap image
            heatmap = cv2.applyColorMap(np.uint8(df_pred), cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(out_filename, heatmap)
            logging.info(f'Distance field saved to {out_filename}')

            # print number of positive pixels in the line map
            line_map[line_map == 1] = 255
            out_overlay_filename = out_overlay_imgs[i]
            cv2.imwrite(out_overlay_filename, line_map)
            logging.info(f'Line map saved to {out_overlay_filename}')
        

        if head_mode == "df_seg":
            df_pred = result[0]
            label_mask_pred = result[1]

            bin_edges = get_bin_edges(net.n_classes)
            line_mask, line_map = df_clsmask_to_linemap(df=df_pred, cls_mask=label_mask_pred, 
                                                        df_neighborhood=10, threshold=0.05,
                                                        bin_edges=bin_edges)
            
            overlay_img = create_pred_vis_overlay(line_map, img, args.scale)
            cv2.imwrite(out_overlay_imgs[i], overlay_img)
            logging.info(f'Overlay image saved to {out_overlay_imgs[i]}')


            # # Normalize the line_map to [0, 255] for heatmap
            # line_map_normalized = cv2.normalize(line_map, None, 0, 255, cv2.NORM_MINMAX)
            # line_map_normalized = np.uint8(line_map_normalized)
            
            # # Generate heatmap from normalized line_map
            # heatmap = cv2.applyColorMap(line_map_normalized, cv2.COLORMAP_HOT)

            # # Resize and center crop the image
            # print("img size: ", img.size)
            # vis_img = img.resize((int(img.width * args.scale), int(img.height * args.scale)), Image.BILINEAR)
            # print("vis_img size: ", vis_img.size)
            # center_crop = transforms.CenterCrop((224, 224))
            # vis_img = center_crop(vis_img)
            # vis_img = np.array(vis_img)

            # # Convert BGR to RGB
            # vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

            # # Overlay heatmap on the original image
            # overlay_img = cv2.addWeighted(vis_img, 0.4, heatmap, 0.6, 0)
            # overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)

            # # Use matplotlib to create the colorbar and combine it with the image
            # fig, ax = plt.subplots(figsize=(8, 8))

            # # Display the overlay image
            # ax.imshow(overlay_img)
            # ax.axis('off')  # Hide axes

            #  # Create a colorbar with the corresponding min/max values
            # cax = fig.add_axes([0.92, 0.25, 0.03, 0.5])  # Position: [left, bottom, width, height]
            # norm = plt.Normalize(vmin=np.min(line_map), vmax=np.max(line_map))
            # sm = plt.cm.ScalarMappable(cmap='hot', norm=norm)
            # sm.set_array([])

            # # Add the colorbar to the image
            # cbar = fig.colorbar(sm, cax=cax)
            # cbar.set_label('Value', rotation=270, labelpad=15)

            # # Add min label to the colorbar
            # cbar.ax.text(1.1, 0, f'{np.min(line_map):.2f}', ha='left', va='bottom', transform=cbar.ax.transAxes)

            # # Save the final image with the colorbar
            # plt.savefig(out_overlay_imgs[i], bbox_inches='tight', pad_inches=0.1)
            # plt.close()
                        

