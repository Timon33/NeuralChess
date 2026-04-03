"""
Evaluation utilities: parsing raw eval strings and scaling centipawn values.
"""

import re

MATE_VALUE = 2000.0
MATE_PATTERN = re.compile(r"^#([+-]?)(\d+)$")


def parse_eval(raw: str | int | float) -> float:
    raw = str(raw).strip()
    match = MATE_PATTERN.match(raw)
    if match:
        sign = -1.0 if match.group(1) == "-" else 1.0
        return sign * MATE_VALUE
    return float(raw)


def scale_eval(cp: float, inverse: bool = False) -> float:
    import torch

    if inverse:
        val = max(min(cp, 0.999999), -0.999999)
        return float(torch.atanh(torch.tensor(val)) * 600.0)
    return float(torch.tanh(torch.tensor(cp / 600.0)))
