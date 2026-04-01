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


def scale_eval(cp: float) -> float:
    import torch

    return float(torch.tanh(torch.tensor(cp / 600.0)))
