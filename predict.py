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
from utils.pred_to_line import df_to_linemap, df_wf_to_linemap

def predict_img(net,
                full_img,
                depth_img,
                device,
                scale_factor = 0.5,
                out_threshold=0.5,
                use_depth=True):
    

    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess( full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    depth = torch.from_numpy(BasicDataset.preprocess(depth_img, scale_factor, is_mask=False, is_depth=True))
    depth = depth.unsqueeze(0)
    depth = depth.to(device=device, dtype=torch.float32)

    
    # do a center crop to both image
    center_crop = transforms.CenterCrop((320, 320))
    img = center_crop(img)
    depth = center_crop(depth)

    imgs = torch.cat((img, depth), dim=1) if use_depth else img

    with torch.no_grad():
        result = net(imgs)        

    return result


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


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


def overlay_img_with_mask(img, mask, alpha=0.5):
    # Convert the image to a numpy array
    overlay_img = np.array(img.copy())
    
    # Define the pink color
    pink = np.array([255, 20, 147], dtype=np.uint8)
    
    # Get the positions where the mask is 1
    positive_pixels = np.where(mask == 1)
    
    # Blend the pink color with the original image
    overlay_img[positive_pixels] = (alpha * pink + (1 - alpha) * overlay_img[positive_pixels]).astype(np.uint8)
    
    return Image.fromarray(overlay_img)



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

    # get all images in in_files folder with .jpg extension, into a list
    print(in_files)
    files = os.listdir(in_files)
    in_imgs = []
    out_imgs = []
    out_overlay_imgs = []

    for f in files:
        if f.endswith('.png') or f.endswith('.jpg'):
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
            # img = np.concatenate([np.array(img), np.array(depth)], axis=-1)
            
        result = predict_img(net=net,
                           full_img=img,
                           depth_img=depth,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device,
                           use_depth=args.use_depth)
        
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
        
        if head_mode == "df_wf":
            df_pred = result[0]
            wf_pred = result[1]

            weighted_map = df_wf_to_linemap(df=df_pred, wf=wf_pred, 
                                            df_neighborhood=10, threshold=1.0)
            
            global_max = 3000.0
            global_min = 1e-6

            # normalize the weighted map using min-max scaling, and use global min and max
            weighted_map = weighted_map*4

            weighted_map = (weighted_map - global_min) / (global_max - global_min) * 255
            # self-normalize the weighted map
            # weighted_map = (weighted_map - weighted_map.min()) / (weighted_map.max() - weighted_map.min()) * 255

            # get a heatmap from the weighted map, 
            # the higher the value, the more bright yellow the pixel
            # the lower the value, the more dark black the pixel
            heatmap = cv2.applyColorMap(np.uint8(weighted_map), cv2.COLORMAP_HOT)

            # overlay the heatmap to the original image
            # first, need to get the same size as the original image, which is first resize and then center crop
            # first, resize using args.scale
            # then, center crop to 320x320
            print("img size: ", img.size)
            vis_img = img.resize((int(img.width * args.scale), int(img.height * args.scale)), Image.BILINEAR)
            # then do numpy center crop
            print("vis_img size: ", vis_img.size)
            center_crop = transforms.CenterCrop((320, 320))
            vis_img = center_crop(vis_img)
            print("vis_img size after center crop: ", vis_img.size)
            print("heatmap size: ", heatmap.shape)
            vis_img = np.array(vis_img)
            # this is currently BGR image, convert to RGB
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
            # overlay the heatmap to the original image
            overlay_img = cv2.addWeighted(vis_img, 0.5, heatmap, 0.5, 0)
            cv2.imwrite(out_overlay_imgs[i], overlay_img)
            logging.info(f'Overlay image saved to {out_overlay_imgs[i]}')
            

            # line_map = df_to_linemap(df_pred, df_neighborhood=10, threshold=0.45)
            # # save the distance field as a heatmap image
            # out_filename = out_imgs[i]
            # # normalize the distance field to 0-255
            # df_pred = df_pred.squeeze().cpu().numpy()
            # df_pred = (df_pred - df_pred.min()) / (df_pred.max() - df_pred.min()) * 255
            # # convert the distance field to a heatmap image
            # heatmap = cv2.applyColorMap(np.uint8(df_pred), cv2.COLORMAP_VIRIDIS)
            # cv2.imwrite(out_filename, heatmap)
            # logging.info(f'Distance field saved to {out_filename}')

            # # print number of positive pixels in the line map
            # line_map[line_map == 1] = 255
            # out_overlay_filename = out_overlay_imgs[i]
            # cv2.imwrite(out_overlay_filename, line_map)
            # logging.info(f'Line map saved to {out_overlay_filename}')

