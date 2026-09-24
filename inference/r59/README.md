# 在线推理提交样例说明

本项目仅用于演示在线推理包的目录、启动方式和结果格式，不是可参赛的声纹模型。`run.py` 当前按文件路径生成固定的伪 embedding，选手必须替换为自己的模型加载和音频推理逻辑。

## 平台约定

- 初赛测试集不向选手开放，由平台在在线推理时挂载。
- 测试集输入路径目前暂定为 `/work/data/占位符`，正式路径确定后以平台公告为准。
- 结果目录为 `/work/output`。
- 程序最终必须生成且只需交付一个结果文件：`/work/output/submit.zip`。
- 本样例也兼容环境变量 `GAME823_INPUT_DIR`、`AIOJ_INPUT_DIR`、`INPUT_DIR` 指定输入目录，以及 `AIOJ_OUTPUT_DIR`、`OUTPUT_DIR` 指定输出目录。

## 必须保持的结果格式

程序需要为每个输入 WAV 生成一个同相对路径、同文件名的 NPZ，只将扩展名从 `.wav` 改为 `.npz`。例如：

```text
输入：data_scenery1/audio/audio_xxx.wav
输出成员：data_scenery1/audio/audio_xxx.npz
```

全部 NPZ 放入单一 `submit.zip`。每个 NPZ 必须满足：

- 包含键 `embedding`；
- `embedding` 是一维、非空、有限实数数组；
- 所有音频的 embedding 维度完全一致，且维度不超过 4096；
- 不得包含 NaN、Inf、复数或零范数向量；
- 单个 NPZ 不超过 16 MiB。

初赛平台会严格核验结果与隐藏输入音频一一对应，缺少、多出、重名或路径错误都会导致评测失败。

## 选手需要修改

1. 将模型权重和配置放入 `model/`。
2. 在 `requirements.txt` 中声明实际依赖，注意平台在线推理环境为 Python 3.8。
3. 替换 `run.py` 中的 `dummy_embedding()`，读取 WAV 并输出真实声纹 embedding。
4. 保留输入路径兼容逻辑、NPZ 路径映射和最终 `submit.zip` 生成逻辑。

只要 `start.sh` 仍通过 `python3 run.py` 启动，通常不需要修改。

## 本地调试

本地可用自备 WAV 模拟平台输入：

```bash
export GAME823_INPUT_DIR=/path/to/local/test
export OUTPUT_DIR=/tmp/game823-output
bash start.sh
```

本地目录仍应包含 `data_scenery1/audio/`、`data_scenery2/audio/` 结构。官方隐藏测试集和 trial 标签不会提供给选手。

## 打包

在本项目目录的上一级执行：

```bash
tar -cvzf game823-speaker-verification-sample.tar.gz game823-speaker-verification-sample
```

上传到 S3/ATP 的是该 `tar.gz` 推理项目包；程序在线运行后生成的是评分回调使用的 `submit.zip`，二者不要混淆。
