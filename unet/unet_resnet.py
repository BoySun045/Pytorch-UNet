import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp

class UnetResnet(nn.Module):
    def __init__(self, n_classes):
        super(UnetResnet, self).__init__()
        self.n_classes = n_classes
        self.model = smp.Unet(
            encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=n_classes               # model output channels (number of classes in your dataset)
        )
        # self.model.to(torch.device("cuda:0"), memory_format=torch.channels_last)

        # output_stride for resnet34 is 32, so need to pad the input image to be divisible by 32
        self.output_stride = 32


    def forward(self, x):
        # do the padding if needed
        h, w = x.shape[-2:]
        if h % self.output_stride != 0 or w % self.output_stride != 0:
            new_h = (h // self.output_stride + 1) * self.output_stride if h % self.output_stride != 0 else h
            new_w = (w // self.output_stride + 1) * self.output_stride if w % self.output_stride != 0 else w
            x = nn.functional.pad(x, (0, new_w - w, 0, new_h - h))
        
        pred = self.model(x)

        # remove the padding if needed
        if h % self.output_stride != 0 or w % self.output_stride != 0:
            pred = pred[:, :, :h, :w]
        return pred