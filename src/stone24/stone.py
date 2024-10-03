from typing import List, Tuple

import gin  # type: ignore
import torch
from torch import Tensor

from src.stone24.model.chromanet import ChromaNet
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
        device: str,
    ) -> None:
        super().__init__()
        self.hcqt = hcqt
        self.device = device
        self.n_harmonics = len(self.hcqt.harmonics)
        self.n_bins = n_bins
        self.bins_before_crop = hcqt.n_bins
        self.out_channels = out_channels
        self.kernels = kernels
        self.chromanet = ChromaNet(self.n_bins, self.n_harmonics, self.out_channels, self.kernels, temperature)
        self.keymode_to_num = {
                 'B minor': 0,
                 'C minor': 1,
                 'C# minor': 2,
                 'D minor': 3,
                 'D# minor': 4,
                 'E minor': 5,
                 'F minor': 6,
                 'F# minor': 7,
                 'G minor': 8,
                 'G# minor': 9,
                 'A minor': 10,
                 'A# minor': 11,
                 'D major': 12,
                 'D# major': 13,
                 'E major': 14,
                 'F major': 15,
                 'F# major': 16,
                 'G major': 17,
                 'G# major': 18,
                 'A major': 19,
                 'A# major': 20,
                 'B major': 21,
                 'C major': 22,
                 'C# major': 23
                 }

    def forward(self, 
                x: dict
                ) -> Tuple[Tensor, Tensor]:
        """
        Return the output for both vocal and accompaniment
        """
        audio = x["audio"]
        assert audio.shape[2] == 1 or audio.shape[2]==2

        # supervised
        if audio.shape[2] == 1:
            batch = audio.shape[0]
            audio = audio.permute(0, 2, 1)
            
            # change string annotation to int
            keymode = [self.keymode_to_num[i] for i in x["keymode"][0]]

            # difference of cropping (positive means pitch down)
            difference = torch.randint(1, 12, (len(audio), ))
            crop_transpose = CropCQT(self.n_bins)
            hcqt = self.hcqt(audio)
            transpose_hcqt = crop_transpose(hcqt, difference)
            hcqt = hcqt[:, :, :84, :]
            stack_input = torch.cat((hcqt, transpose_hcqt), dim=0)

            # calculate output of chromanet
            y = self.chromanet(stack_input)
            # ground truth of y from annotations
            y_gt = torch.zeros((batch, 24)).to(self.device)
            for i, idx in enumerate(keymode):
                y_gt[i, idx] = 1
            y = torch.cat((y_gt, y), dim=0)
        
        # self-supervised, same as stone12
        else:
            s1 = audio[:, :, 0] # batch, channel, n_bins, time
            s2 = audio[:, :, 1]
            batch = audio.shape[0]
            stack_hcqt = self.hcqt(torch.cat((s1, s2), dim=0))

            # calculate the transposition values, all is done to make sure that variable "difference" is of the full range [-11, 11]
            to_transpose = torch.randint(1, 12, (len(s1), )) # always positive
            original = torch.randint(1, 13, (len(s1), )) # pitch to transpose segment 1 and 2
            transpose = (to_transpose+original) % 12 # pitch to transpose segment 1, always positive
            difference = transpose - original 
            crop_fn = CropCQT(self.n_bins)

            # crop CQT
            stack_original = crop_fn(stack_hcqt, torch.cat((original, original), dim=0))
            segment_transpose = crop_fn(stack_hcqt[:batch, ...], transpose)
            stack_input = torch.cat((stack_original, segment_transpose), dim=0) # torch.Size([384, 1, 84, 646])
            y = self.chromanet(stack_input) # (384, 24)

        assert y.shape[1] == 24

        return (y, difference)
