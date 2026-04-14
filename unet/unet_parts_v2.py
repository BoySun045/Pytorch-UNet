import torch
import torch.nn as nn
import torch.nn.functional as F

from segmentation_models_pytorch.base import modules as md

import torch.nn as nn
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.base import initialization as init


class DetrHead(nn.Module):
    """DETR-style sparse set prediction head.

    Predicts N slot vectors via cross-attention of learned queries to
    multi-scale memory tokens built from UNet encoder features (bottleneck
    + x3, pooled to the same spatial size).

    Returns per slot:
        uv   : sigmoid → (B, N, 2)  normalised pixel coords in [0, 1]
        z    : softplus → (B, N, 1) metric depth in metres
        conf : sigmoid → (B, N, 1) slot-active probability
    """

    def __init__(
        self,
        bottleneck_channels: int,
        x3_channels: int,
        num_queries: int = 10,
        d_model: int = 256,
        nhead: int = 8,
        num_decoder_layers: int = 3,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model

        # Project encoder feature maps to a common d_model dimension
        self.proj_bottleneck = nn.Linear(bottleneck_channels, d_model)
        self.proj_x3 = nn.Linear(x3_channels, d_model)

        # Learnable object queries (à la DETR)
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Transformer decoder: queries cross-attend to memory tokens
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=d_model * 4,
            dropout=0.1,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Per-slot prediction heads
        self.head_uv     = nn.Linear(d_model, 2)   # sigmoid  → [0, 1] normalised
        self.head_z      = nn.Linear(d_model, 1)   # softplus → depth [m]
        self.head_conf   = nn.Linear(d_model, 1)   # sigmoid  → slot confidence
        self.head_weight = nn.Linear(d_model, 1)   # softplus → GMM component weight
        self.head_occ    = nn.Linear(d_model, 1)   # sigmoid  → occlusion probability

    # ------------------------------------------------------------------
    def _build_memory(self, features):
        """Flatten + project bottleneck and x3 encoder maps into memory tokens.

        Pools x3 (H/8) to match bottleneck (H/32) spatial size so token
        counts are equal; final memory has 2·S tokens.
        """
        bottleneck = features[-1]   # (B, C_bn, H/32, W/32)
        x3         = features[-3]   # (B, C_x3, H/8,  W/8 )

        # Pool x3 to bottleneck spatial size
        x3_pooled = F.adaptive_avg_pool2d(x3, bottleneck.shape[2:])   # (B, C_x3, H/32, W/32)

        # Flatten spatial → (B, S, C)
        bn_flat = bottleneck.flatten(2).transpose(1, 2)   # (B, S, C_bn)
        x3_flat = x3_pooled.flatten(2).transpose(1, 2)    # (B, S, C_x3)

        # Project to d_model
        memory = torch.cat(
            [self.proj_bottleneck(bn_flat), self.proj_x3(x3_flat)], dim=1
        )   # (B, 2S, d_model)
        return memory

    def forward(self, features):
        memory = self._build_memory(features)                           # (B, S, d_model)
        B = memory.shape[0]

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, N, d_model)
        slots   = self.transformer_decoder(queries, memory)               # (B, N, d_model)

        uv     = torch.sigmoid(self.head_uv(slots))      # (B, N, 2)
        z      = F.softplus(self.head_z(slots))          # (B, N, 1)
        conf   = torch.sigmoid(self.head_conf(slots))    # (B, N, 1)
        weight = F.softplus(self.head_weight(slots))     # (B, N, 1)  unnormalised weight
        occ    = torch.sigmoid(self.head_occ(slots))     # (B, N, 1)  occlusion prob

        return uv, z, conf, weight, occ


class ScaledTanh(nn.Module):
    def __init__(self):
        super(ScaledTanh, self).__init__()

    def forward(self, x):
        return 0.5 * (torch.tanh(x) + 1.0)
    

class ClampReLU(nn.Module):
    def __init__(self):
        super(ClampReLU, self).__init__()

    def forward(self, x):
        return F.relu(x).clamp(min=0, max=1)
    
class PredictionModel(torch.nn.Module):
    def initialize(self,head_config, df_neighborhood):
        init.initialize_decoder(self.decoder)
        if head_config == "both":
            init.initialize_head(self.segmentation_head)
            init.initialize_head(self.regression_head)
        elif head_config == "segmentation":
            init.initialize_head(self.segmentation_head)
        elif head_config == "regression":
            init.initialize_head(self.regression_head)
        elif head_config == "df_wf":
            init.initialize_head(self.df_regression_head)
            init.initialize_head(self.wf_regression_head)
        elif head_config == "df_seg":
            init.initialize_head(self.segmentation_head)
            init.initialize_head(self.df_regression_head)
        elif head_config == "detr":
            # DetrHead uses standard PyTorch init; init aux depth head if present
            if getattr(self, "detr_aux_depth", False):
                init.initialize_head(self.aux_depth_head)

        self.head_mode = head_config
        self.df_neighborhood = df_neighborhood

    def check_input_shape(self, x):

        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        # do the padding if needed
        h, w = x.shape[-2:]
        if h % self.output_stride != 0 or w % self.output_stride != 0:
            new_h = (h // self.output_stride + 1) * self.output_stride if h % self.output_stride != 0 else h
            new_w = (w // self.output_stride + 1) * self.output_stride if w % self.output_stride != 0 else w
            x = nn.functional.pad(x, (0, new_w - w, 0, new_h - h))

        features = self.encoder(x)

        # DETR sparse prediction from encoder features
        if self.head_mode == "detr":
            uv, z, conf, weight, occ = self.detr_head(features)
            # Auxiliary dense depth head: run the UNet decoder when enabled
            if self.detr_aux_depth:
                decoder_output = self.decoder(*features)
                depth_pred = self.aux_depth_head(decoder_output)  # [B, 1, H, W]
                # Remove padding added at input
                depth_pred = depth_pred[:, :, :h, :w]
            else:
                depth_pred = None
            return uv, z, conf, weight, occ, depth_pred

        decoder_output = self.decoder(*features)

        if self.head_mode == "both":
            masks = self.segmentation_head(decoder_output)
            values = self.regression_head(decoder_output)         
            # remove the padding if needed
            if h % self.output_stride != 0 or w % self.output_stride != 0:
                masks = masks[:, :, :h, :w]
                values = values[:, :, :int(h * self.regression_head.downsample_factor), :int(w * self.regression_head.downsample_factor)]
            return masks, values            

        elif self.head_mode == "segmentation":
            masks = self.segmentation_head(decoder_output)
            if h % self.output_stride != 0 or w % self.output_stride != 0:
                masks = masks[:, :, :h, :w]
            return masks
        
        elif self.head_mode == "regression":
            values = self.regression_head(decoder_output)
            if h % self.output_stride != 0 or w % self.output_stride != 0:
                 values = values[:, :, :int(h * self.regression_head.downsample_factor), :int(w * self.regression_head.downsample_factor)]
            return values
        
        elif self.head_mode == "df":
            norm_values = self.regression_head(decoder_output)
            if h % self.output_stride != 0 or w % self.output_stride != 0:
                 norm_values = norm_values[:, :, :int(h * self.regression_head.downsample_factor), :int(w * self.regression_head.downsample_factor)]
            
            return norm_values

        elif self.head_mode == "df_wf":
            df = self.df_regression_head(decoder_output)
            wf = self.wf_regression_head(decoder_output)

            if h % self.output_stride != 0 or w % self.output_stride != 0:
                 df = df[:, :, :int(h * self.df_regression_head.downsample_factor), :int(w * self.df_regression_head.downsample_factor)]
                 wf = wf[:, :, :int(h * self.wf_regression_head.downsample_factor), :int(w * self.wf_regression_head.downsample_factor)]
            
            return df, wf
        
        elif self.head_mode == "df_seg":
            masks = self.segmentation_head(decoder_output)
            df = self.df_regression_head(decoder_output)

            if h % self.output_stride != 0 or w % self.output_stride != 0:
                masks = masks[:, :, :h, :w]
                df = df[:, :, :int(h * self.df_regression_head.downsample_factor), :int(w * self.df_regression_head.downsample_factor)]

            return df, masks

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        # conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        # upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        # activation = Activation(activation)
        # super().__init__(conv2d, upsampling, activation)

        conv2d_1 = nn.Conv2d(in_channels, 64, kernel_size=kernel_size, padding=kernel_size // 2)
        activation_1 = nn.ReLU()
        batch_norm_1 = nn.BatchNorm2d(64)
        conv2d_2 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=kernel_size // 2)
        activation_2 = nn.ReLU()
        batch_norm_2 = nn.BatchNorm2d(64)
        conv2d_3 = nn.Conv2d(64, out_channels, kernel_size=1)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation_3 = Activation(activation)
        super().__init__(conv2d_1, activation_1, batch_norm_1, conv2d_2, activation_2, batch_norm_2, conv2d_3, upsampling, activation_3)

        

class RegressionHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, 
                 downsample_factor, 
                 kernel_size=3, activation=None, upsampling=1):
        self.downsample_factor = downsample_factor

        conv2d_1 = nn.Conv2d(in_channels, 64, kernel_size=kernel_size, padding=kernel_size // 2)
        activation_1 = nn.ReLU()
        # batch_norm_1 = nn.BatchNorm2d(64)
        conv2d_2 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=kernel_size // 2)
        activation_2 = nn.ReLU()
        # batch_norm_2 = nn.BatchNorm2d(64)
        conv2d_3 = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Downsampling layer
        downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=int(1/downsample_factor), padding=1) if downsample_factor != 1 else nn.Identity()
        
        # activation_3 = ScaledTanh()  
        activation_3 = nn.ReLU()
        
        # super().__init__(conv2d_1, activation_1, batch_norm_1, conv2d_2, activation_2, batch_norm_2, conv2d_3, downsample, activation_3)
        super().__init__(conv2d_1, activation_1, conv2d_2, activation_2, conv2d_3, activation_3)

class DfRegressionHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, 
                downsample_factor, 
                kernel_size=3, activation=None, upsampling=1):
        
        self.downsample_factor = downsample_factor

        conv2d_1 = nn.Conv2d(in_channels, 64, kernel_size=kernel_size, padding=kernel_size // 2)
        activation_1 = nn.ReLU()
        batch_norm_1 = nn.BatchNorm2d(64)
        conv2d_2 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=kernel_size // 2)
        activation_2 = nn.ReLU()
        batch_norm_2 = nn.BatchNorm2d(64)
        conv2d_3 = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Downsampling layer
        downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=int(1/downsample_factor), padding=1) if downsample_factor != 1 else nn.Identity()
        activation_3 = nn.ReLU()  
        super().__init__(conv2d_1, activation_1, batch_norm_1, conv2d_2, activation_2, batch_norm_2, conv2d_3, downsample, activation_3)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_norm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
