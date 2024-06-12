from typing import Optional, Union, List

from segmentation_models_pytorch.encoders import get_encoder

from .unet_parts_v2 import (
    UnetDecoder,
    RegressionHead,
    PredictionModel,
    SegmentationHead
)
import torch.nn as nn
import torch 


class TwoHeadUnet(PredictionModel):
    """Unet_ is a fully convolution neural network for image semantic segmentation. Consist of *encoder*
    and *decoder* parts connected with *skip connections*. Encoder extract features of different spatial
    resolution (skip connections) which are used by decoder to define accurate segmentation mask. Use *concatenation*
    for fusing decoder blocks with skip connections.

    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features
            two times smaller in spatial dimensions than previous one (e.g. for depth 0 we will have features
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and
            other pretrained weights (see table with available weights for each encoder_name)
        decoder_channels: List of integers which specify **in_channels** parameter for convolutions used in decoder.
            Length of the list should be the same as **encoder_depth**
        decoder_use_batchnorm: If **True**, BatchNorm2d layer between Conv2D and Activation layers
            is used. If **"inplace"** InplaceABN will be used, allows to decrease memory consumption.
            Available options are **True, False, "inplace"**
        decoder_attention_type: Attention module used in decoder of the model. Available options are
            **None** and **scse** (https://arxiv.org/abs/1808.08127).
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Available options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"tanh"**, **"identity"**,
                **callable** and **None**.
            Default is **None**
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax"
                    (could be **None** to return logits)

    Returns:
        ``torch.nn.Module``: Unet

    .. _Unet:
        https://arxiv.org/abs/1505.04597

    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        head_config: str = "both",  # "both", "segmentation", "regression"
        regression_downsample_factor: float = 1.0 
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        if in_channels == 4:
            # Modify the first convolutional layer to accept 4 channels
            self.encoder.conv1 = self.modify_first_conv(self.encoder.conv1, in_channels)

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        if head_config == "both":
            self.segmentation_head = SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=classes,
                activation=activation,
                kernel_size=3,
            )

            self.regression_head = RegressionHead(
                in_channels=decoder_channels[-1],
                out_channels=classes,
                downsample_factor=regression_downsample_factor,
                activation=activation, # always use relu in the regression class definition
                kernel_size=5,
            )

        if head_config == "segmentation":
            self.segmentation_head = SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=classes,
                activation=activation,
                kernel_size=3,
            )

            self.regression_head = None


        if head_config == "regression":
            self.segmentation_head = None

            self.regression_head = RegressionHead(
                in_channels=decoder_channels[-1],
                out_channels=classes,
                downsample_factor=regression_downsample_factor,
                activation=activation, # always use relu in the regression class definition
                kernel_size=5,
            )

        self.name = "u-{}".format(encoder_name)
        self.initialize(head_config)

        self.output_stride = 32
        self.n_classes = classes
        self.n_channels = in_channels

    @staticmethod
    def modify_first_conv(conv_layer, in_channels):
        if in_channels == conv_layer.in_channels:
            return conv_layer
        else:
            new_conv = nn.Conv2d(
                in_channels, conv_layer.out_channels,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.padding,
                bias=(conv_layer.bias is not None)
            )

            # Adjust weights for new in_channels
            with torch.no_grad():
                if in_channels < conv_layer.in_channels:
                    new_conv.weight.data = conv_layer.weight.data[:, :in_channels, :, :]
                else:
                    new_conv.weight.data[:, :conv_layer.in_channels, :, :] = conv_layer.weight.data
                    new_conv.weight.data[:, conv_layer.in_channels:, :, :] = conv_layer.weight.data[:, :in_channels - conv_layer.in_channels, :, :]

            return new_conv