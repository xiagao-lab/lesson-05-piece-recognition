# 虾搞象棋第 5 课：整盘棋子识别

本课只解决一件事：把第 4 课拉正的棋盘图，变成棋子位置与种类表。

```text
board.png → 一次 ONNX 推理 → [1, 90, 16] → 10×9 布局 → board.json
```

这里不是把棋盘切成 90 张图再调用 90 次模型。`90 = 10×9` 个交叉点，`16 = 空位、其他、7 种红子、7 种黑子`；模型一次看整盘，并为每个位置输出 16 个分数。

## 模型从哪里来

本项目使用上游仓库提供的预训练模型 `nano_v3-0319.onnx`，不是本课程自行训练的模型，也不在本仓库重复分发：

- <https://github.com/TheOne1006/chinese-chess-recognition>
- <https://huggingface.co/spaces/yolo12138/Chinese_Chess_Recognition/tree/main/onnx>

下载后保存为 `models/classifier.onnx`。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_models.py
python lesson/run.py --image /path/to/board.png
```

输出包括：

- `outputs/lesson05/layout.txt`：10 行 × 9 列的紧凑布局。
- `outputs/lesson05/board.json`：棋子种类与 `row/col` 坐标。
- `outputs/lesson05/recognition.json`：基础质量检查。

技术选择和输出契约见 [docs/TECHNOLOGY.md](docs/TECHNOLOGY.md)。

## 课后练习

仓库不提供标准答案。完成 `exercises/starter.py` 后运行：

```bash
python exercises/check.py
```

使用 [exercises/AI_PRACTICE.md](exercises/AI_PRACTICE.md) 可以让 AI 逐步提示，而不是直接生成答案。

## 课程材料

- `courseware/`：带提词器备注和内嵌视频的 PPT。
- `visual-demos/`：整盘一次推理、90×16 输出与棋子表的动态演示。
- `src/xiagao_chess/recognition.py`：与演示核对过的实际后处理代码。
