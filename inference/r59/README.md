# 多编码器球面融合推理运行时

本目录是最佳模型的推理运行时。模型权重不存放在 Git 仓库中；请先按照仓库根目录的 [README](../../README.md) 下载并安装完整 `weights` 目录。

推荐从仓库根目录运行统一入口：

```bash
python scripts/infer.py \
  --input /path/to/input \
  --output outputs/inference
```

输入目录应包含 `data_sceneryN/audio/*.wav`。程序为每条音频生成一个包含 `embedding` 键的 `float32[4096]` NPZ 文件，并写入 `outputs/inference/submit.zip`。

也可以直接运行比赛格式入口：

```bash
export GAME823_INPUT_DIR=/path/to/input
export OUTPUT_DIR=/path/to/output
bash inference/r59/start.sh
```

该运行时已经包含真实的多编码器声纹模型加载与推理实现，无需修改 `run.py`。
