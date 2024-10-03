from typing import Any, Tuple
import torch
from torch.utils.data import IterableDataset
import gin  # type: ignore


@gin.configurable
class ToyDataset(IterableDataset):  
    """
    A toy dataset which feeds data of shape that the model needs. Users should modify it to output real audio data. 
    """
    def __init__(
            self, 
            batch_size: int, 
            sr: int, 
            duration: int,
            device: str,
            dataset_type: str, # the type of dataset.
    ):
        """
        Parameters:
            batch_size: batch size of the data.
            sr: sampling rate of audios, 22050Hz is used in the original paper.
            duration: duration of each segment, 15s is used in the original paper.
            device: device where the model is trained.
            dataset_type: mix, supervised or selfsupervised. While supervised, the audio shape is (batch_size, sr*duration, 1), the keymode is a tuple of a list of labels. While selfsupervised, the audio shape is (batch_size, sr*duartion, 2), 2 for two segments, the keymode is a tuple of a list of string "-1". While mixed, the dataloader alternates between supervised and selfsupervised batches.
        """
        assert dataset_type in ["mix", "supervised", "selfsupervised"], "Invalid dataset_type!"
        self.batch_size = batch_size
        self.duration = duration
        self.sr = sr
        self.audio_shape = (sr * duration, 2) if dataset_type=="supervised" else (sr*duration, 1)
        self.dataset_type = dataset_type
        self.device = device
        self.iter_count = 0


    def __iter__(self):
        while True:

            if self.dataset_type == "supervised":
                audio = torch.randn(self.batch_size, *self.audio_shape)
                keymode = (["A minor"]*self.batch_size,)

            elif self.dataset_type == "selfsupervised":
                audio = torch.randn(self.batch_size, *self.audio_shape)
                keymode = (["-1"]*self.batch_size,)

            elif self.dataset_type == "mix":
                # Alternate between supervised and self-supervised
                if self.iter_count % 2 == 0:
                    audio = torch.randn(self.batch_size, *(self.sr * self.duration, 1)).to(self.device)
                    keymode = (["A minor"] * self.batch_size,)
                else:
                    audio = torch.randn(self.batch_size, *(self.sr * self.duration, 2)).to(self.device)
                    keymode = (["-1"] * self.batch_size,)

                self.iter_count += 1

            # Yield the batch dictionary
            yield {
                "audio": audio,
                "keymode": keymode
            }


@gin.configurable  # type: ignore
def get_datasets(
        device: str,
        dataset_type_train: str,
        dataset_type_test: str,
        batch_size: int,
        sr: int,
        duration: int,
) -> Tuple[Any, Any]:

    ds_train = ToyDataset(batch_size, sr, duration, device, dataset_type_train)
    ds_test = ToyDataset(batch_size, sr, duration, device, dataset_type_test)


    return ds_train, ds_test
