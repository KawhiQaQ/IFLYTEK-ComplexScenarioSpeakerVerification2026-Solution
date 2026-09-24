# IFLYTEK Complex-Scenario Speaker Verification 2026

Solution for the “Voiceprint in the Mist” Complex-Scenario Speaker Verification Challenge（“声纹迷雾”复杂场景说话人确认挑战赛）

[中文](README.md) · [Method](docs/METHOD.md) · [Model Weights](https://pan.baidu.com/s/1Jz5MlxJOKvoUdZkTl5n9oA?pwd=qndz)

## Overview

This repository contains the training and inference code for our final competition solution. The task extracts a speaker representation from each utterance and uses cosine similarity to verify whether enrollment and test speech belong to the same speaker. The evaluation covers child speech, same-gender interference, near/far-field conditions, device mismatch, and 0.5- and 1-second utterances.

The final model produces a 4,096-dimensional embedding and achieved **93.40746** on the semifinal public leaderboard. Inference uses only the current waveform; it does not use filenames, trial pairs, scenario labels, thresholds, or test-set statistics.

## Method

We use a **multi-encoder sphere-fusion speaker model** with four main components:

- **Complementary acoustic encoders:** ERes2Net, CAM++, ResNet293, ReDimNet2, and W2V-BERT 2.0 provide complementary speaker, channel, distance, and short-duration cues.
- **Wide residual speaker head:** zero-initialized wide residual adapters operate on the frozen W2V-BERT 2.0 layer representations, followed by depth-time interaction.
- **Cross-encoder residual:** ReDimNet2 frames serve as keys and values while SSL frames serve as queries, producing an acoustic-to-SSL residual correction.
- **Unit-sphere fusion:** branch embeddings are normalized independently and concatenated with fixed weights; the original GRL head and cross-encoder head are averaged on the unit sphere.

Task training uses 2,077 speakers, including 317 child speakers, random durations of 0.5, 1, 2, and 3 seconds, and classification, prototype, hard-negative, completion, duration-consistency, and relation-preservation objectives. See [METHOD.md](docs/METHOD.md) for the architecture and losses.

## Reproduction

### 1. Environment

The recommended environment is x86-64 Linux with an NVIDIA GPU, Python 3.8, PyTorch 2.4.1, and CUDA 12.1.

```bash
git clone https://github.com/KawhiQaQ/IFLYTEK-ComplexScenarioSpeakerVerification2026-Solution.git
cd IFLYTEK-ComplexScenarioSpeakerVerification2026-Solution
bash scripts/bootstrap.sh
source .conda/envs/miwu/bin/activate
```

### 2. Model Weights

Download the complete `weights` directory from [Baidu Netdisk](https://pan.baidu.com/s/1Jz5MlxJOKvoUdZkTl5n9oA?pwd=qndz), extraction code `qndz`, and place it in the repository root:

```text
IFLYTEK-ComplexScenarioSpeakerVerification2026-Solution/
├── weights/
├── scripts/
├── src/
└── ...
```

Install and verify the model assets:

```bash
python scripts/install_weights.py --weights weights
python scripts/verify_assets.py
```

SHA-256 of the final online model archive:

```text
415856babca0285f63c052acbd74e79a23c46e688bd708d5f5341137efc1c7a3
```

### 3. Inference with Released Weights

Input audio must be 16 kHz, 16-bit, mono PCM WAV in the competition layout:

```text
/path/to/input/
└── data_scenery1/
    └── audio/
        ├── a.wav
        └── b.wav
```

Run inference:

```bash
python scripts/infer.py \
  --input /path/to/input \
  --output outputs/inference
```

The command writes `outputs/inference/submit.zip`. Each WAV maps to an NPZ at the same relative path, with a finite `float32[4096]` array under the `embedding` key.

### 4. Training

Prepare the training corpora using the following layout:

```text
data/
├── processed/
│   ├── 3dspeaker/
│   ├── commonvoice17-train/
│   ├── speechocean/
│   └── stcmds/ST-CMDS-20170001_1-OS/
└── raw/
    └── childmandarin/train/
```

Validate all paths and print the training plan:

```bash
python scripts/train.py --dry-run
```

Run the complete task-training pipeline and build the inference package:

```bash
python scripts/train.py
```

The default command rebuilds training-identity prototypes, trains the wide residual speaker head, trains the cross-encoder residual, and assembles the final model. Training starts from the public pretrained acoustic encoders distributed with the weights; foundation-model pretraining from random initialization is outside the scope of this solution.

## Repository Layout

```text
configs/       training configuration
deployment/    final sphere-fusion module
docs/          method description
inference/     inference runtime
scripts/       environment, weight installation, training, and inference
src/miwu/      model and data implementation
vendor/        third-party source and licenses
```

## License

Original code is released under the [MIT License](LICENSE). Third-party code, datasets, and pretrained weights retain their respective licenses and terms of use.
