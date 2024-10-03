from typing import Any, Callable, Dict, Union

import torch
import gin
import torch.nn as nn


log_clap: Callable[[torch.Tensor], torch.Tensor] = lambda x: torch.clamp(
    torch.log(x), min=-100
)

class Z_transformation(torch.nn.Module):
    """
    Complex Z tranformation for loss calculation, project the 12 probability bins to a single point on a disk of r=1. 
    """
    def __init__(self,
                 circle_type: int,
                 device: str) -> None:
        """
        Params:
            circle_type: 1 or 7. 1 represents the circle of semitones, 7 represents the circle of fifths.
            device: the device where z transformation is calculated.
        """
        super().__init__()
        self.omega = circle_type/12
        self.alpha = torch.exp(1j * 2 * torch.pi * self.omega * torch.arange(12))
        self.device = device

    def forward(self,
                y: torch.Tensor
                ) -> torch.Tensor:
        z = torch.matmul(torch.complex(y, 0*y), self.alpha.cuda(self.device)) # equation 2 in the original paper.
        return z


@gin.configurable
class CrossPowerSpectralDensityLoss(nn.Module):
    """
    Differentialble distance on the circle of fifths.
    """
    def __init__(self, 
                 circle_type: int,
                 device: str,
             ) -> None:
        """
        Params:
            circle_type: 1 or 7. 1 represents the circle of fifths, 7 represents the circle of semitones.
            device: the device where z transformation is calculated.
        """
        super(CrossPowerSpectralDensityLoss, self).__init__()
        self.z_transformation = Z_transformation(circle_type, device)


    def forward(
        self, 
        y: torch.Tensor
    ) -> Dict[str, Union[int, float, Dict]]:
        y, difference = y
        batch_size = int(y.shape[0]/3)

        # calculate m, value for mode, vertical summation
        channel_1, channel_2 = torch.split(y, 12, dim=1) # [n_segments*batch+equivariant, 12]
        m1 = torch.sum(channel_1, dim=1) # sum of mode per data point in the batch, equation 14 in the original paper

        # for 3 views
        m1_source1 = m1[:batch_size]
        m1_source2 = m1[batch_size:2*batch_size]
        m1_equivariant = m1[2*batch_size:]

        # horizontal summation
        y = torch.add(channel_1, channel_2) # equation 13 in the original paper

        # z transformation, for 3 views
        z = self.z_transformation(y)
        z1 = z[:batch_size, ...]
        z2 = z[batch_size: batch_size*2, ...]
        z3 = z[batch_size*2:, ...]

        # loss calculation
        loss_pos = (1 - z1 * z2.conj()).abs().pow(2).mean()
        loss_equivariant_1 = (torch.exp(1j * 2 * torch.pi * self.z_transformation.omega * difference.to(self.z_transformation.device)) - z1 * z3.conj()).abs().pow(2).mean() # equation 5 in the original paper
        loss_equivariant_2 = (torch.exp(1j * 2 * torch.pi * self.z_transformation.omega * difference.to(self.z_transformation.device)) - z2 * z3.conj()).abs().pow(2).mean()
        loss_key = loss_pos + loss_equivariant_1 + loss_equivariant_2

        loss_mode = (
                (-m1_source1*log_clap(m1_equivariant)-(1-m1_source1)*log_clap(1-m1_equivariant)).mean() + 
                0.5*(-m1_source2*log_clap(m1_equivariant)-(1-m1_source2)*log_clap(1-m1_equivariant)).mean() + 
                0.5*(-m1_equivariant*log_clap(m1_source2)-(1-m1_equivariant)*log_clap(1-m1_source2)).mean() + 
                (-m1_source1*log_clap(m1_source2)-(1-m1_source1)*log_clap(1-m1_source2)).mean()
                ) # cross entropy on the two colomns
        loss = loss_key + 1.5*loss_mode

        return {"loss": loss, "loss_to_print": {"loss_pos": loss_pos, "loss_equi": loss_equivariant_1+loss_equivariant_2, "loss_mode": loss_mode, "loss_total":loss}}
