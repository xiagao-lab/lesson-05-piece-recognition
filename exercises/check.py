from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("exercise05", HERE / "starter.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def main() -> int:
    logits = np.zeros((1, 90, 16), dtype=np.float32)
    logits[..., 0] = 1.0
    logits[0, 4, 9] = 5.0
    logits[0, 85, 2] = 6.0
    layout = module.decode_logits(logits)
    lines = layout.splitlines()
    assert len(lines) == 10 and all(len(line) == 9 for line in lines)
    assert lines[0] == "....k...."
    assert lines[9] == "....K...."
    assert module.generals_are_sane(layout) is True
    assert module.generals_are_sane(layout.replace("K", ".")) is False
    print("练习 05 通过：一次输出已还原为 10×9 布局，并完成基础质量检查。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
