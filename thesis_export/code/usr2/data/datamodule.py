from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Grayscale,
    Lambda,
    Resize,
)

from .dataset import AVDataset
from .samplers import ByFrameCountSampler, DistributedSamplerWrapper, RandomSamplerWrapper
from .transforms import AddNoise, NormalizeVideo


def pad(samples, pad_val=0.0):
    if len(samples) == 0:
        return None, None

    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) < 3:
        collated_batch = collated_batch.unsqueeze(1)
    else:
        collated_batch = collated_batch.permute((0, 4, 1, 2, 3))
    return collated_batch, lengths


def collate_fn(batch):
    batch_out = {}
    for data_type in ('video', 'audio', 'label'):
        pad_val = -1 if data_type == 'label' else 0.0
        c_batch, sample_lengths = pad([s[data_type] for s in batch if s[data_type] is not None], pad_val)
        batch_out[data_type] = c_batch
        batch_out[data_type + '_lengths'] = sample_lengths

    batch_out["path"] = [s["path"] for s in batch if s["path"] is not None]

    return batch_out


class USRDataModule(LightningDataModule):

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.total_gpus = self.cfg.gpus * self.cfg.trainer.num_nodes

    def _video_transform(self):
        args = self.cfg.data
        transform = [
            Lambda(lambda x: x / 255.),
            CenterCrop(args.crop_type.random_crop_dim),
            Resize(args.crop_type.resize_dim),
        ]

        if self.cfg.data.channel.in_video_channels == 1:
            transform.extend([
                Lambda(lambda x: x.transpose(0, 1)),
                Grayscale(),
                Lambda(lambda x: x.transpose(0, 1)),
            ])
        transform.append(NormalizeVideo(args.channel.obj.mean, args.channel.obj.std))

        return Compose(transform)

    def _audio_transform(self):
        args = self.cfg.data
        transform = [Lambda(lambda x: x)]
        if args.noise_path is not None:
            transform.append(
                AddNoise(
                    noise_path=args.noise_path,
                    snr_target=getattr(self.cfg.decode, "snr_target", 9999),
                )
            )
        return Compose(transform)

    def _dataloader(self, ds, sampler):
        return DataLoader(
            ds,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            batch_sampler=sampler,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        ds_args = self.cfg.data.dataset

        transform_video = self._video_transform()
        transform_audio = self._audio_transform()

        test_ds = AVDataset(
            data_path=ds_args.test_csv,
            video_path_prefix_lrs2=self.cfg.data.lrs2_video_dir,
            audio_path_prefix_lrs2=self.cfg.data.lrs2_audio_dir,
            video_path_prefix_lrs3=self.cfg.data.lrs3_video_dir,
            audio_path_prefix_lrs3=self.cfg.data.lrs3_audio_dir,
            video_path_prefix_vox2=self.cfg.data.vox2_video_dir,
            audio_path_prefix_vox2=self.cfg.data.vox2_audio_dir,
            video_path_prefix_avsp=self.cfg.data.avsp_video_dir,
            audio_path_prefix_avsp=self.cfg.data.avsp_audio_dir,
            transforms={'video': transform_video, 'audio': transform_audio},
        )
        sampler = ByFrameCountSampler(test_ds, self.cfg.data.frames_per_gpu_val, shuffle=False)
        sampler = RandomSamplerWrapper(sampler)
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=False)

        return [self._dataloader(test_ds, sampler)]
