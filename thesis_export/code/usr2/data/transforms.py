import random

import numpy as np
import torch


def normalize(clip, mean, std, inplace=False):
    assert clip.ndimension() == 4, "clip should be a 4D torch.tensor"
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip


class NormalizeVideo:
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        return normalize(clip, self.mean, self.std, self.inplace)


class AddNoise:
    def __init__(self, noise_path, snr_target=None, snr_levels=[-5, 0, 5, 10, 15, 20, 9999]):
        self.noise = np.load(noise_path)
        self.snr_levels = snr_levels
        self.snr_target = snr_target

    def get_power(self, clip):
        return np.sum(clip ** 2) / (len(clip) * 1.0)

    def __call__(self, signal):
        device = signal.device
        signal = signal[0].numpy()
        snr_target = random.choice(self.snr_levels) if self.snr_target is None else self.snr_target
        if snr_target == 9999:
            return torch.tensor(signal, device=device)
        else:
            start_idx = random.randint(0, len(self.noise) - len(signal))
            noise_clip = self.noise[start_idx:start_idx + len(signal)]
            sig_power = self.get_power(signal)
            noise_clip_power = self.get_power(noise_clip)
            factor = (sig_power / noise_clip_power) / (10 ** (snr_target / 10.0))
            desired_signal = (signal + noise_clip * np.sqrt(factor)).astype(np.float32)
            return torch.unsqueeze(torch.tensor(desired_signal, device=device), 0)
