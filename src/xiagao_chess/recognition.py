from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


CLASS_NAMES = (
    "point",
    "other",
    "red_king",
    "red_advisor",
    "red_bishop",
    "red_knight",
    "red_rook",
    "red_cannon",
    "red_pawn",
    "black_king",
    "black_advisor",
    "black_bishop",
    "black_knight",
    "black_rook",
    "black_cannon",
    "black_pawn",
)

SHORT_LABELS = (".", "x", "K", "A", "B", "N", "R", "C", "P", "k", "a", "b", "n", "r", "c", "p")

PIECE_NAME = {
    "K": "redGeneral",
    "A": "redAdvisor",
    "B": "redElephant",
    "N": "redHorse",
    "R": "redChariot",
    "C": "redCannon",
    "P": "redSoldier",
    "k": "blackGeneral",
    "a": "blackAdvisor",
    "b": "blackElephant",
    "n": "blackHorse",
    "r": "blackChariot",
    "c": "blackCannon",
    "p": "blackSoldier",
}

PIECE_LIMITS = {
    "K": 1,
    "k": 1,
    "A": 2,
    "a": 2,
    "B": 2,
    "b": 2,
    "N": 2,
    "n": 2,
    "R": 2,
    "r": 2,
    "C": 2,
    "c": 2,
    "P": 5,
    "p": 5,
}


@dataclass(frozen=True)
class RecognitionResult:
    layout: str
    selected_scores: np.ndarray
    board: list[dict]
    sane: bool
    count_rules_ok: bool
    mean_selected_score: float

    def to_dict(self) -> dict:
        return {
            "layout": self.layout,
            "board": self.board,
            "quality": {
                "has_one_red_and_black_general": self.sane,
                "piece_count_rules_ok": self.count_rules_ok,
                "mean_selected_model_score": round(self.mean_selected_score, 6),
            },
        }


class BoardClassifier:
    """整盘一次推理，输出 90 个位置 × 16 个类别。"""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"找不到棋子识别模型: {self.model_path}")
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    @staticmethod
    def _preprocess(board_rgb: np.ndarray) -> np.ndarray:
        height, width = board_rgb.shape[:2]
        crop_w, crop_h = 400, 450
        if width < crop_w or height < crop_h:
            board_rgb = cv2.resize(board_rgb, (max(width, crop_w), max(height, crop_h)))
            height, width = board_rgb.shape[:2]
        x1 = (width - crop_w) // 2
        y1 = (height - crop_h) // 2
        cropped = board_rgb[y1 : y1 + crop_h, x1 : x1 + crop_w]
        resized = cv2.resize(cropped, (280, 315))
        normalized = (
            resized.astype(np.float32) - np.array([123.675, 116.28, 103.53], dtype=np.float32)
        ) / np.array([58.395, 57.12, 57.375], dtype=np.float32)
        return np.expand_dims(np.transpose(normalized, (2, 0, 1)), axis=0)

    def predict(self, board_rgb: np.ndarray) -> RecognitionResult:
        tensor = self._preprocess(board_rgb)
        outputs = self.session.run(None, {self.input_name: tensor})
        if len(outputs) != 1 or outputs[0].shape[1:] != (90, 16):
            raise RuntimeError(f"识别模型输出应为 [batch,90,16]，实际为 {[o.shape for o in outputs]}")
        logits = outputs[0][0]
        indices = np.argmax(logits, axis=1)
        selected = logits[np.arange(90), indices]
        chars = [SHORT_LABELS[int(index)] for index in indices]
        rows = [chars[index * 9 : (index + 1) * 9] for index in range(10)]
        layout = "\n".join("".join(row) for row in rows)
        board = layout_to_board(layout)
        sane = layout.count("K") == 1 and layout.count("k") == 1
        counts_ok = all(layout.count(char) <= limit for char, limit in PIECE_LIMITS.items())
        return RecognitionResult(
            layout=layout,
            selected_scores=selected.reshape(10, 9),
            board=board,
            sane=sane,
            count_rules_ok=counts_ok,
            mean_selected_score=float(np.mean(selected)),
        )


def layout_to_board(layout: str) -> list[dict]:
    lines = [line.strip() for line in layout.splitlines() if line.strip()]
    if len(lines) != 10 or any(len(line) != 9 for line in lines):
        raise ValueError("棋盘布局必须恰好为 10 行 × 9 列")
    result: list[dict] = []
    for row_index, line in enumerate(lines):
        detector_row = 10 - row_index
        for col_index, char in enumerate(line):
            if char in PIECE_NAME:
                result.append(
                    {
                        "piece": PIECE_NAME[char],
                        "cord": {"row": detector_row, "col": col_index + 1},
                    }
                )
    return result
