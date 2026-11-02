---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
dataset_info:
  features:
  - name: idx
    dtype: int64
  - name: audio
    sequence: int16
  - name: video
    sequence:
      sequence:
        sequence: uint8
  - name: label
    dtype: string
  splits:
  - name: train
    num_bytes: 824374107
    num_examples: 1321
  download_size: 677311360
  dataset_size: 824374107
---
# Dataset Card for "lrs3-test"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)