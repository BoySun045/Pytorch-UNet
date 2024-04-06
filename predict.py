import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', help='Filenames of input images', required=True, type=str)
    parser.add_argument('--output', '-o', metavar='OUTPUT', help='Filenames of output images')
    parser.add_argument('--save_overlay_output', '-so', metavar='SAVE_OVERLAY', help='Filenames of overlay images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--use_depth', '-d', action='store_true', help='Use depth images')

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

    if args.use_depth:
        # depth file is in the same directory as the image file but with a different name, depth
        depth_files = in_files.replace('image', 'depth')

    if not os.path.exists(out_files):
        os.makedirs(out_files)
    if not os.path.exists(out_overlay_files):
        os.makedirs(out_overlay_files)

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)

    logging.info('Model loaded!')

    # get all images in in_files folder with .jpg extension, into a list
    print(in_files)
    files = os.listdir(in_files)
    in_imgs = []
    out_imgs = []
    out_overlay_imgs = []

    for f in files:
        if f.endswith('.jpg'):
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
            depth = Image.open(depth_filename)
            img = np.concatenate([np.array(img), np.array(depth)], axis=-1)
            
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        # overlay the mask on the image
        overlay_img = overlay_img_with_mask(img, mask)

        if not args.no_save:
            out_filename = out_imgs[i]
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            out_overlay_filename = out_overlay_imgs[i]
            overlay_img.save(out_overlay_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
