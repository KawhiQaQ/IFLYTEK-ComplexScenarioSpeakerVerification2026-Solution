<h1 align="center">IFLYTEK ComplexScenarioSpeakerVerification2026: Solution</h1>

<p align="center"><strong>“声纹迷雾”复杂场景说话人确认挑战赛</strong></p>

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

## 简介

本仓库开源比赛最终方案的训练与推理代码。任务要求为每条语音生成说话人表征，并通过余弦相似度判断注册语音与测试语音是否来自同一说话人。评测覆盖儿童、同性干扰、远近场、设备差异以及 0.5 秒和 1 秒超短语音等复杂条件。

## 方案介绍

我们采用**多编码器球面融合说话人模型**，核心包括：

- **互补声学编码器**：融合 ERes2Net、CAM++、ResNet293、ReDimNet2 与 W2V-BERT 2.0 的说话人表征，覆盖音色、设备、距离和短时信息。
- **宽残差说话人头**：在冻结的 W2V-BERT 2.0 编码器上，对各层表示加入零初始化宽残差适配器，并进行深度—时间交互。
- **跨编码器残差**：使用 ReDimNet2 帧作为键和值、SSL 帧作为查询，学习声学特征到自监督特征的残差修正。
- **单位球面融合**：各分支先独立归一化，再以固定权重拼接；原始 GRL 头与跨编码器头在单位球面上等权合成。

任务训练使用 2077 个说话人，其中 317 个儿童说话人。训练采用 0.5、1、2、3 秒随机时长，以及分类、原型、困难负样本、短语音补全、时长一致性和关系保持目标。完整结构与损失定义见 [METHOD.md](docs/METHOD.md)。

## 复现指南

### 1. 环境配置

推荐环境为 x86-64 Linux、NVIDIA GPU、Python 3.8、PyTorch 2.4.1 和 CUDA 12.1。

```bash
git clone https://github.com/KawhiQaQ/IFLYTEK-ComplexScenarioSpeakerVerification2026-Solution.git
cd IFLYTEK-ComplexScenarioSpeakerVerification2026-Solution
bash scripts/bootstrap.sh
source .conda/envs/miwu/bin/activate
```

### 2. 下载权重

从[百度网盘](https://pan.baidu.com/s/1Jz5MlxJOKvoUdZkTl5n9oA?pwd=qndz)下载完整 `weights` 目录，提取码：`qndz`。将其放在仓库根目录：

```text
IFLYTEK-ComplexScenarioSpeakerVerification2026-Solution/
├── weights/
├── scripts/
├── src/
└── ...
```

安装并校验模型资产：

```bash
python scripts/install_weights.py --weights weights
python scripts/verify_assets.py
```

最终线上模型归档的 SHA-256 为：

```text
415856babca0285f63c052acbd74e79a23c46e688bd708d5f5341137efc1c7a3
```

### 3. 直接加载权重推理

输入为 16 kHz、16-bit、单声道 PCM WAV，并保持比赛目录结构：

```text
/path/to/input/
└── data_scenery1/
    └── audio/
        ├── a.wav
        └── b.wav
```

运行推理：

```bash
python scripts/infer.py \
  --input /path/to/input \
  --output outputs/inference
```

输出为 `outputs/inference/submit.zip`。每个 WAV 对应一个同相对路径的 NPZ 文件，其中 `embedding` 为有限的 `float32[4096]` 向量。

### 4. 自行训练

准备以下训练数据目录：

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

先检查数据、权重和训练命令：

```bash
python scripts/train.py --dry-run
```

按论文方案的默认设置完成全部任务训练并生成推理包：

```bash
python scripts/train.py
```

该命令依次重建训练身份原型、训练宽残差说话人头、训练跨编码器残差，并组装最终模型。这里的自行训练指从随权重包提供的公开预训练声学编码器开始完整运行任务训练；不包含基础声学模型的随机初始化预训练。

## 仓库结构

```text
configs/       训练配置
deployment/    最终球面融合模块
docs/          方法说明
inference/     推理运行时
scripts/       环境、权重安装、训练和推理入口
src/miwu/      模型与数据实现
vendor/        第三方依赖源码与许可证
```

## 许可证

自研代码使用 [MIT License](LICENSE)。第三方代码、数据和预训练权重遵循各自的许可证与使用条款。
