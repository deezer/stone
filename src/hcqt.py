import math
import gin
import nnAudio.features.vqt
import torch
import torchaudio


# relies on nnAudio v0.3.2
# pip install git+https://github.com/KinWaiCheuk/nnAudio.git#subdirectory=Installation
@gin.configurable
class HarmonicVQT(nnAudio.features.vqt.VQT):
    r"""Harmonic VQT: A collection of VQTs with different shifts.
    Inspired by Bittner, McFee, Salamon, Li, Bello.
    "Deep Salience Representations for F0 Estimation in Polyphonic Music". ISMIR 2017

    Args:
        harmonics (Collection[float]): Harmonics to be included.
        fmin (float): Minimum frequency to be included.
        n_bins (int): Number of bins in the output spectrogram.
        bins_per_octave (int, optional): Number of bins per octave. Defaults to 12.
    """
    def __init__(self, *, harmonics, fmin, n_bins, bins_per_octave=12, **kwargs):
        self.harmonics = harmonics
        self.bin_shifts = []
        self.n_bins_per_slice = n_bins
        self.fmin = fmin
        for harmonic in harmonics:
            shift = round(bins_per_octave * math.log2(harmonic))
            self.bin_shifts.append(shift)
        low_octave_shift = min([0] + self.bin_shifts) / bins_per_octave
        fmin = fmin * (2 ** low_octave_shift)
        n_bins = n_bins + max([0] + self.bin_shifts) - min([0] + self.bin_shifts)
        super().__init__(fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave, **kwargs)


    def forward(self, x, output_format='Magnitude', normalization_type='librosa'):
        vqt = super().forward(x, output_format, normalization_type)
        hvqt = []
        for shift in self.bin_shifts:
            bin_start = shift - min([0] + self.bin_shifts)
            bin_stop = bin_start + self.n_bins_per_slice
            vqt_slice = vqt[:, bin_start:bin_stop, ...]
            hvqt.append(vqt_slice)
        hvqt = torch.stack(hvqt, dim=1)
        log_hcqt = ((1.0/80.0) * torchaudio.transforms.AmplitudeToDB(top_db=80)(hvqt)) + 1.0
        return log_hcqt


class CropCQT(torch.nn.Module):
    def __init__(self, height: int):
        super(CropCQT, self).__init__()
        self.height = height 

    def forward(self, spectrograms: torch.Tensor, transpose: torch.Tensor) -> torch.Tensor:
        return torch.stack([s[:, int(l):int(l) + self.height, :] for s, l in zip(spectrograms, transpose)])
