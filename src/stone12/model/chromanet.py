from typing import List
from torch import nn
from src.stone12.model.convnext import ConvNeXtBlock, TimeDownsamplingBlock
from einops import rearrange
import gin


class OctavePool(nn.Module):
    r"""Average log-frequency axis across octaves, thus producing a chromagram."""
    def __init__(self, bins_per_octave):
        super().__init__()
        self.bins_per_octave = bins_per_octave

    def forward(self, x):
        # x: (batch_size, channel, H, W)
        x = rearrange(x, "B C (j k) W -> B C k j W", k=self.bins_per_octave)
        x = x.mean(dim=3)
        return x


@gin.configurable
class ChromaNet(nn.Module):
    def __init__(self, 
                 n_bins: int,
                 n_harmonics: int, 
                 out_channels: List[int],
                 kernels: List[int],
                 temperature: float,
                 ):
        super().__init__()
        assert len(kernels) == len(out_channels)
        self.n_harmonics = n_harmonics
        self.n_bins = n_bins
        in_channel = self.n_harmonics
        self.out_channels = out_channels
        self.kernels = kernels
        self.temperature = temperature
        self.drop_path = 0.1
        convnext_blocks = []
        time_downsampling_blocks = []
        for i, out_channel in enumerate(self.out_channels):
            time_downsampling_block = TimeDownsamplingBlock(in_channel, out_channel, n_bins)
            kernel = self.kernels[i]
            convnext_block = ConvNeXtBlock(out_channel, out_channel, n_bins, kernel_size=kernel, padding=kernel//2, drop_path = self.drop_path)
            time_downsampling_blocks.append(time_downsampling_block)
            convnext_blocks.append(convnext_block)
            in_channel = out_channel
        self.convnext_blocks = nn.ModuleList(convnext_blocks)
        self.time_downsampling_blocks = nn.ModuleList(time_downsampling_blocks)
        self.octave_pool = OctavePool(12)
        self.global_average_pool = nn.AdaptiveAvgPool2d((12, 1))
        self.classifier = nn.Conv2d(out_channel, 1, kernel_size=(1, 1))
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        block_zip = zip(self.convnext_blocks, self.time_downsampling_blocks)
        for convnext_block, time_downsampling_block in block_zip:
            x = time_downsampling_block(x)
            x = convnext_block(x)
        x = self.octave_pool(x)
        x = self.global_average_pool(x)
        x = self.classifier(x)
        x = self.flatten(x)
        x = self.softmax(x/self.temperature)
        assert x.shape[1] == 12
        return x
