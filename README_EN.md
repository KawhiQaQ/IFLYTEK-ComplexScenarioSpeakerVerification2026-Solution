<h1 align="center">IFLYTEK ComplexScenarioSpeakerVerification2026: Solution</h1>

<p align="center"><strong>“Voiceprint in the Mist” Complex-Scenario Speaker Verification Challenge</strong></p>

<p align="center">
  <a href="README_EN.md">English</a> |
  <a href="README.md">简体中文</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8-3776AB?logo=python&logoColor=white" alt="Python 3.8">
  <img src="https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.4.1">
  <img src="https://img.shields.io/badge/Semifinal%20LB-93.40746-7B3FC6" alt="Semifinal leaderboard score 93.40746">
  <img src="https://img.shields.io/badge/Embedding-4096D-2EA44F" alt="4096-dimensional embedding">
  <img src="https://img.shields.io/badge/License-MIT-1683BB" alt="MIT License">
</p>

---

## Overview

This repository contains the training and inference code for our final competition solution. The task extracts a speaker representation from each utterance and uses cosine similarity to verify whether enrollment and test speech belong to the same speaker. The evaluation covers child speech, same-gender interference, near/far-field conditions, device mismatch, and 0.5- and 1-second utterances.

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
