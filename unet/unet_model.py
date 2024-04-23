""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        # Paths for regression
        self.up1_reg = Up(1024, 512 // factor, bilinear)
        self.up2_reg = Up(512, 256 // factor, bilinear)
        self.up3_reg = Up(256, 128 // factor, bilinear)
        self.up4_reg = Up(128, 64, bilinear)
        self.outc_reg = OutConv(64, n_classes, activation="tanh")

        # Paths for binary classification
        self.up1_bin = Up(1024, 512 // factor, bilinear)
        self.up2_bin = Up(512, 256 // factor, bilinear)
        self.up3_bin = Up(256, 128 // factor, bilinear)
        self.up4_bin = Up(128, 64, bilinear)
        self.outc_bin = OutConv(64, n_classes, activation="sigmoid")
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Regression path
        x_reg = self.up1_reg(x5, x4)
        x_reg = self.up2_reg(x_reg, x3)
        x_reg = self.up3_reg(x_reg, x2)
        x_reg = self.up4_reg(x_reg, x1)
        logits_reg = self.outc_reg(x_reg)

        # Binary classification path
        x_bin = self.up1_bin(x5, x4)
        x_bin = self.up2_bin(x_bin, x3)
        x_bin = self.up3_bin(x_bin, x2)
        x_bin = self.up4_bin(x_bin, x1)
        logits_bin = self.outc_bin(x_bin)

        return logits_reg, logits_bin

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        
        self.up1_reg = torch.utils.checkpoint(self.up1_reg)
        self.up2_reg = torch.utils.checkpoint(self.up2_reg)
        self.up3_reg = torch.utils.checkpoint(self.up3_reg)
        self.up4_reg = torch.utils.checkpoint(self.up4_reg)
        self.outc_reg = torch.utils.checkpoint(self.outc_reg)

        self.up1_bin = torch.utils.checkpoint(self.up1_bin)
        self.up2_bin = torch.utils.checkpoint(self.up2_bin)
        self.up3_bin = torch.utils.checkpoint(self.up3_bin)
        self.up4_bin = torch.utils.checkpoint(self.up4_bin)
        self.outc_bin = torch.utils.checkpoint(self.outc_bin)

        