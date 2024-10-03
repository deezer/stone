from typing import Dict, Union

import torch
import torch.nn as nn


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


class CrossPowerSpectralDensityLoss(nn.Module):
    """
    Differentialble distance on the circle of fifths.
    """
    def __init__(self, 
                 circle_type: int,
                 device: str
             ) -> None:
        super(CrossPowerSpectralDensityLoss, self).__init__()
        self.z_transformation = Z_transformation(circle_type, device)

    def forward(
        self, 
        y: torch.Tensor
    ) -> Dict[str, Union[int, float, Dict]]:
        y, difference = y
        batch_size = int(y.shape[0]/3)
        z = self.z_transformation(y)
        z1 = z[:batch_size, ...] # segment 1
        z2 = z[batch_size: batch_size*2, ...] # segment 2
        z3 = z[batch_size*2:, ...] # transposed segment 1
        loss_pos = (1 - z1 * z2.conj()).abs().pow(2).mean()
        loss_equivariant_1 = (torch.exp(1j * 2 * torch.pi * self.z_transformation.omega * difference.to(self.z_transformation.device)) - z1 * z3.conj()).abs().pow(2).mean()
        loss_equivariant_2 = (torch.exp(1j * 2 * torch.pi * self.z_transformation.omega * difference.to(self.z_transformation.device)) - z2 * z3.conj()).abs().pow(2).mean()
        loss = loss_pos + loss_equivariant_1 + loss_equivariant_2
        return {"loss": loss, "loss_to_print": {"loss_pos": loss_pos, "loss_equi": loss_equivariant_1+loss_equivariant_2, "loss_total":loss}}
