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

### 4. Data Preparation

Training uses only the subsets listed below. Review and comply with each dataset's license and terms before downloading or using it.

| Dataset | Required download | Source | Final path |
| --- | --- | --- | --- |
| 3D-Speaker | `test.tar.gz` and `3dspeaker_files.tar.gz` | [Project page](https://3dspeaker.github.io/) | `data/processed/3dspeaker/` |
| SpeechOcean762 | Complete archive | [OpenSLR 101](https://www.openslr.org/101/) | `data/processed/speechocean/` |
| ST-CMDS | `ST-CMDS-20170001_1-OS` | [OpenSLR 38](https://www.openslr.org/38/) | `data/processed/stcmds/ST-CMDS-20170001_1-OS/` |
| ChildMandarin | `new_data/train.tar` | [Hugging Face](https://huggingface.co/datasets/BAAI/ChildMandarin) | `data/raw/childmandarin/train/` |
| Common Voice 17.0 | All 26 `validation` Parquet shards for `zh-CN` | [17.0 dataset card](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0), [Mozilla Data Collective](https://mozilladatacollective.com/datasets) | `data/raw/commonvoice17-zhcn-validation/` |

Download and extract the three directly accessible public datasets:

```bash
mkdir -p data/downloads data/processed/3dspeaker data/processed/stcmds

curl -L https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/3D-Speaker/test.tar.gz \
  -o data/downloads/3dspeaker-test.tar.gz
curl -L https://speech-lab-share-data.oss-cn-shanghai.aliyuncs.com/3D-Speaker/3dspeaker_files.tar.gz \
  -o data/downloads/3dspeaker_files.tar.gz
tar -xzf data/downloads/3dspeaker-test.tar.gz -C data/processed/3dspeaker
tar -xzf data/downloads/3dspeaker_files.tar.gz -C data/processed/3dspeaker

curl -L https://www.openslr.org/resources/101/speechocean762.tar.gz \
  -o data/downloads/speechocean762.tar.gz
tar -xzf data/downloads/speechocean762.tar.gz -C data/processed
mv data/processed/speechocean762 data/processed/speechocean

curl -L https://www.openslr.org/resources/38/ST-CMDS-20170001_1-OS.tar.gz \
  -o data/downloads/ST-CMDS-20170001_1-OS.tar.gz
tar -xzf data/downloads/ST-CMDS-20170001_1-OS.tar.gz \
  -C data/processed/stcmds
```

ChildMandarin is gated. Accept its terms on the dataset page, authenticate, and download only the training archive:

```bash
huggingface-cli login
huggingface-cli download BAAI/ChildMandarin new_data/train.tar \
  --repo-type dataset --local-dir data/downloads/childmandarin
mkdir -p data/raw/childmandarin
tar -xf data/downloads/childmandarin/new_data/train.tar \
  -C data/raw/childmandarin
```

Common Voice must be release **17.0**, locale **`zh-CN`**, split **`validation`**. A newer release will not reproduce the fixed split. Place `validation_0.parquet` through `validation_25.parquet` in `data/raw/commonvoice17-zhcn-validation/`, then build the released 600-speaker training subset:

```bash
python scripts/prepare_commonvoice17.py \
  --root data/raw/commonvoice17-zhcn-validation \
  --output-root data/processed/commonvoice17-train \
  --split-root outputs/commonvoice17
```

The resulting layout must be:

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

### 5. Training

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
