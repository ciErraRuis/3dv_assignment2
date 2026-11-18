"""Explicit controlled variants rather than an uncontrolled Cartesian product."""

from __future__ import annotations

import copy
from pathlib import Path

from .config import PipelineConfig
from .errors import ValidationError
from .io import content_hash, exact_keys, read_json, safe_id, write_json


def merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            # Changing a component replaces its old constructor options.
            if "name" in value and value["name"] != result[key].get("name"):
                result[key] = copy.deepcopy(value)
            else:
                result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class ExperimentSuite:
    def __init__(self, specification: dict) -> None:
        exact_keys(
            specification,
            {"schema_version", "question", "controls", "limitations", "base", "variants"},
            {"schema_version", "question", "controls", "base", "variants"},
            "suite",
        )
        if specification["schema_version"] != 1 or not specification["variants"]:
            raise ValidationError("Expected schema_version=1 and nonempty variants")
        self.specification = specification

    @classmethod
    def load(cls, path: Path) -> ExperimentSuite:
        return cls(read_json(path))

    def expand(self) -> tuple[PipelineConfig, ...]:
        configs, names = [], set()
        for variant in self.specification["variants"]:
            exact_keys(variant, {"name", "overrides"}, {"name", "overrides"}, "variant")
            name = safe_id(variant["name"])
            if name in names:
                raise ValidationError(f"Duplicate experiment variant: {name}")
            names.add(name)
            raw = merge(self.specification["base"], variant["overrides"])
            raw["name"] = name
            configs.append(PipelineConfig.parse(raw))
        return tuple(configs)

    def write_plan(self, output: Path) -> None:
        configs = self.expand()
        output.mkdir(parents=True, exist_ok=False)
        for config in configs:
            write_json(output / f"{config.name}.json", config)
        write_json(
            output / "plan.json",
            {
                "status": "DRAFT_NOT_EXECUTED",
                "question": self.specification["question"],
                "controls": self.specification["controls"],
                "limitations": self.specification.get("limitations", []),
                "variants": [{"name": c.name, "config_sha256": content_hash(c)} for c in configs],
                "input_policy": "identity RGB only; real-data preprocessing and budgets still need review",
            },
        )
