from typing import Any, Tuple, Union
import gin  # type: ignore
import torch
from torch.utils.data import IterableDataset


class ToyDataset(IterableDataset):
    """
    Create a Pytorch dataset that can generate infinite amount of data points, with a shape of (batch_size, sr*duration, 2). 
    2 corresponds to the number of segments as input to the model. Both segments come from the same audio.
    """
    def __init__(self, sr, duration, batch_size, device):
        self.batch_size = batch_size 
        self.data_shape = (sr*duration, 2)
        self.sr = sr
        self.duration = duration
        self.device = device

    def __iter__(self):
        while True:
            yield torch.randn(self.batch_size, *self.data_shape).to(self.device)

@gin.configurable  
def get_datasets(
        batch_size: int,
        sr: int,
        duration: int,
        device: str,
) -> Tuple[Any, Any]:

    ds_train = ToyDataset(sr, duration, batch_size, device)
    ds_test = ToyDataset(sr, duration, batch_size, device)
    
    return ds_train, ds_test

