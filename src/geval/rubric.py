"""YAML rubric parser and validator."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from .models import DimensionDef, Rubric


def load_rubric(path: Union[str, Path]) -> Rubric:
    """Parse a YAML rubric file and return a validated Rubric."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    dimensions = [DimensionDef(**d) for d in raw["dimensions"]]
    return Rubric(name=raw["name"], dimensions=dimensions)
