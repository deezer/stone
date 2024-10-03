from typing import List, Tuple

import gin  # type: ignore
import torch
from torch import Tensor

from src.stone12.model.chromanet import ChromaNet
from src.hcqt import HarmonicVQT, CropCQT


@gin.configurable
class Stone(torch.nn.Module):
    def __init__(
        self,
        hcqt: HarmonicVQT,
        out_channels: List[int],
        kernels: List[int],
        temperature: float,
        n_bins: int,
    ) -> None:
        super().__init__()
        self.hcqt = hcqt
        self.n_harmonics = len(self.hcqt.harmonics)
        self.n_bins = n_bins
        self.bins_before_crop = hcqt.n_bins
        self.out_channels = out_channels
        self.kernels = kernels
        self.chromanet = ChromaNet(self.n_bins, self.n_harmonics, self.out_channels, self.kernels, temperature)

    def forward(self, 
                x: Tensor
                ) -> Tuple[Tensor, Tensor]:
        """
        Return the output for both vocal and accompaniment
        """
        # format data
        s1 = x[:, :, 0] # batch, channel, n_bins, time
        s2 = x[:, :, 1]
        batch = x.shape[0]
        stack_hcqt = self.hcqt(torch.cat((s1, s2), dim=0))

        # calculate the transposition values, all is done to make sure that variable "difference" is of the full range [-11, 11]
        to_transpose = torch.randint(1, 12, (len(s1), )) # always positive
        original = torch.randint(1, 13, (len(s1), )) # pitch to transpose segment 1 and 2
        transpose = (to_transpose+original) % 12 # pitch to transpose segment 1, always positive
        difference = transpose - original 

        # crop CQT
        crop_fn = CropCQT(self.n_bins)
        stack_original = crop_fn(stack_hcqt, torch.cat((original, original), dim=0))
        segment_transpose = crop_fn(stack_hcqt[:batch, ...], transpose)
        stack_input = torch.cat((stack_original, segment_transpose), dim=0) # torch.Size([384, 1, 84, 646])

        y = self.chromanet(stack_input) # (384, 12)
        assert y.shape[1] == 12

        return (y, difference)

