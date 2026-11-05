from __future__ import annotations

import numpy as np


SHORT_LABELS = (".", "x", "K", "A", "B", "N", "R", "C", "P", "k", "a", "b", "n", "r", "c", "p")


def decode_logits(logits: np.ndarray) -> str:
    """把 [1,90,16] 或 [90,16] 分数转成 10 行 × 9 列布局。"""
    raise NotImplementedError("TODO: 校验形状、argmax、每 9 个字符换一行")


def generals_are_sane(layout: str) -> bool:
    """红帅 K 与黑将 k 必须各出现一次。"""
    raise NotImplementedError("TODO: count 大写 K 与小写 k")
