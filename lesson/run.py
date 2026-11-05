from __future__ import annotations

import argparse
import json
from pathlib import Path

from xiagao_chess.image_io import load_rgb, save_json
from xiagao_chess.recognition import BoardClassifier


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="第 5 课：整盘棋子识别")
    parser.add_argument("--image", type=Path, required=True, help="第 4 课输出的 board.png")
    parser.add_argument("--model", type=Path, default=ROOT / "models/classifier.onnx")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lesson05")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = BoardClassifier(args.model).predict(load_rgb(args.image))
    (args.output / "layout.txt").write_text(result.layout + "\n", encoding="utf-8")
    save_json(args.output / "board.json", result.board)
    save_json(args.output / "recognition.json", result.to_dict())
    print(result.layout)
    print(json.dumps(result.to_dict()["quality"], ensure_ascii=False, indent=2))
    print(f"输出目录: {args.output}")
    return 0 if result.sane else 2


if __name__ == "__main__":
    raise SystemExit(main())
