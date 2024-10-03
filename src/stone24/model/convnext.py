import torch
from torch import nn

class DropPath(nn.Module):
    r"""Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class ConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, in_channels, out_channels, kernel_size=7, padding=3, drop_path=0.1, layer_scale_init_value=1e-1):
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, padding_mode="replicate"
        )  # depthwise conv
        self.norm = nn.functional.layer_norm
        self.pwconv1 = nn.Linear(
            out_channels, 4 * out_channels
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * out_channels, in_channels)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x, x.shape[1:])
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class TimeDownsamplingBlock(nn.Module):
    r"""Time Downsampling Block: LayerNorm -> 1x2 strided Conv -> GELU."""
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.norm = nn.functional.layer_norm
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 2), stride=(1, 2), bias=bias
        )
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x, x.shape[1:])
        x = self.conv(x)
        x = self.act(x)
        return x
