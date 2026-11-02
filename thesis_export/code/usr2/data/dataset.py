import os

import cv2
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


def cut_or_pad(data, size, dim=0):
    if data.size(dim) < size:
        padding = size - data.size(dim)
        data = torch.from_numpy(np.pad(data, (0, padding), "constant"))
    elif data.size(dim) > size:
        data = data[:size]
    return data


class AVDataset(Dataset):
    def __init__(
        self,
        data_path,
        video_path_prefix_lrs2,
        audio_path_prefix_lrs2,
        video_path_prefix_lrs3,
        audio_path_prefix_lrs3,
        video_path_prefix_vox2=None,
        audio_path_prefix_vox2=None,
        video_path_prefix_avsp=None,
        audio_path_prefix_avsp=None,
        transforms=None,
    ):
        self.data_path = data_path
        self.video_path_prefix_lrs3 = video_path_prefix_lrs3
        self.audio_path_prefix_lrs3 = audio_path_prefix_lrs3
        self.video_path_prefix_lrs2 = video_path_prefix_lrs2
        self.audio_path_prefix_lrs2 = audio_path_prefix_lrs2
        self.video_path_prefix_vox2 = video_path_prefix_vox2
        self.audio_path_prefix_vox2 = audio_path_prefix_vox2
        self.video_path_prefix_avsp = video_path_prefix_avsp
        self.audio_path_prefix_avsp = audio_path_prefix_avsp
        self.transforms = transforms

        self.samples = self._load_manifest()

    def _load_manifest(self):
        samples = []
        with open(self.data_path, "r") as f:
            for line in f.read().splitlines():
                tag, file_path, count, label = line.split(",")
                samples.append((tag, file_path, int(count),
                                [int(x) for x in label.split()]))
        return samples

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if not frames:
            return None

        frames = torch.from_numpy(np.stack(frames))
        frames = frames.permute((3, 0, 1, 2))
        return frames

    def load_audio(self, path):
        audio, sr = torchaudio.load(path, normalize=True)
        audio = audio.squeeze(0)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        return audio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        tag, file_path, count, label = self.samples[index]

        video_path_prefix = getattr(self, f"video_path_prefix_{tag}", "")
        audio_path_prefix = getattr(self, f"audio_path_prefix_{tag}", "")

        video = self.load_video(os.path.join(video_path_prefix, file_path))
        if video is None:
            return {'video': None, 'audio': None, 'label': None, 'path': None}

        audio = self.load_audio(os.path.join(audio_path_prefix, file_path[:-4] + ".wav"))
        target_len = video.size(1) * 640
        audio = cut_or_pad(audio, target_len)
        audio = self.transforms["audio"](audio.unsqueeze(0)).squeeze(0)

        video = self.transforms["video"](video).permute((1, 2, 3, 0))

        return {
            "video": video,
            "audio": audio,
            "label": torch.tensor(label),
            "path": file_path,
        }
