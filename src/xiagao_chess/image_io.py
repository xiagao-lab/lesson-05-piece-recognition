from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_rgb(path: str | Path) -> np.ndarray:
    """按 EXIF 方向读取图片，统一返回 RGB ndarray。"""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image)


def save_rgb(path: str | Path, image_rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb.astype(np.uint8), mode="RGB").save(path)


def save_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def draw_text(image_rgb: np.ndarray, text: str, origin: tuple[int, int]) -> np.ndarray:
    result = image_rgb.copy()
    cv2.putText(
        result,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return result
