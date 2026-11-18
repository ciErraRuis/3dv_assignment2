"""Training-view selection never changes the held-out set."""

from __future__ import annotations

import math
from typing import Sequence

from .errors import ValidationError
from .interfaces import ViewSelector
from .types import View


class AllViews(ViewSelector):
    def select(self, views: Sequence[View]) -> tuple[View, ...]:
        return tuple(views)


class FractionSelector(ViewSelector):
    """Index sampling is explicitly not a claim about angular coverage."""

    def __init__(self, fraction: float = 0.5, strategy: str = "uniform_index") -> None:
        if type(fraction) not in (float, int) or not 0 < fraction <= 1:
            raise ValidationError("fraction must be in (0, 1]")
        if strategy not in {"uniform_index", "contiguous", "angular_farthest"}:
            raise ValidationError(f"Unknown selection strategy: {strategy}")
        self.fraction, self.strategy = fraction, strategy

    def select(self, views: Sequence[View]) -> tuple[View, ...]:
        train = [v for v in views if v.split == "train"]
        if not train:
            raise ValidationError("No training views to select")
        count = max(1, math.ceil(len(train) * self.fraction))
        if self.strategy == "uniform_index":
            indices = (
                [round(i * (len(train) - 1) / (count - 1)) for i in range(count)]
                if count > 1
                else [0]
            )
        elif self.strategy == "contiguous":
            indices = list(range(count))
        else:
            if any(v.azimuth_deg is None for v in train):
                raise ValidationError(
                    "angular_farthest requires measured/provided azimuth_deg for every training view"
                )
            indices = [min(range(len(train)), key=lambda i: train[i].azimuth_deg % 360)]

            def distance(i, j):
                delta = abs(train[i].azimuth_deg - train[j].azimuth_deg) % 360
                return min(delta, 360 - delta)

            while len(indices) < count:
                remaining = [i for i in range(len(train)) if i not in indices]
                indices.append(
                    max(remaining, key=lambda i: (min(distance(i, j) for j in indices), -i))
                )
        chosen = {train[i].id for i in indices}
        return tuple(v for v in views if v.split != "train" or v.id in chosen)


class PairGenerator:
    def __init__(self, strategy: str = "exhaustive", window: int = 5) -> None:
        self.strategy, self.window = strategy, window

    def generate(self, count: int):
        if self.strategy not in {"exhaustive", "sequential"} or self.window < 1:
            raise ValidationError("Invalid pair generation configuration")
        for i in range(count):
            stop = count if self.strategy == "exhaustive" else min(count, i + self.window + 1)
            for j in range(i + 1, stop):
                yield i, j
