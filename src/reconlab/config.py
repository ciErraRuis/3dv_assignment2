"""Strict JSON configuration, without heavyweight modeling dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .errors import ValidationError
from .io import exact_keys, read_json, safe_id


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    options: dict = field(default_factory=dict)

    @classmethod
    def parse(cls, value: dict) -> ComponentSpec:
        exact_keys(value, {"name", "options"}, {"name"}, "component")
        if not isinstance(value["name"], str) or not isinstance(value.get("options", {}), dict):
            raise ValidationError("Component requires a string name and object options")
        return cls(**value)


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    selector: ComponentSpec = field(default_factory=lambda: ComponentSpec("all"))
    extractor: ComponentSpec = field(default_factory=lambda: ComponentSpec("sift"))
    matcher: ComponentSpec = field(default_factory=lambda: ComponentSpec("ratio"))
    verifier: ComponentSpec = field(default_factory=lambda: ComponentSpec("fundamental"))
    pose_estimator: ComponentSpec = field(default_factory=lambda: ComponentSpec("hloc"))
    reconstructor: ComponentSpec = field(default_factory=lambda: ComponentSpec("splatfacto"))
    preprocessor: ComponentSpec = field(default_factory=lambda: ComponentSpec("identity"))
    pairing: str = "exhaustive"
    pair_window: int = 5
    seed: int = 42
    geometry_scope: str = "all_selected_views_transductive"
    plugins: tuple[str, ...] = ()

    @classmethod
    def parse(cls, data: dict) -> PipelineConfig:
        allowed = set(cls.__dataclass_fields__)
        exact_keys(data, allowed, {"name"}, "pipeline")
        data = dict(data)
        safe_id(data["name"])
        for key in (
            "selector",
            "extractor",
            "matcher",
            "verifier",
            "pose_estimator",
            "reconstructor",
            "preprocessor",
        ):
            if key in data:
                data[key] = ComponentSpec.parse(data[key])
        config = cls(**data)
        if config.pairing not in {"exhaustive", "sequential"}:
            raise ValidationError("pairing must be exhaustive or sequential")
        if type(config.pair_window) is not int or config.pair_window < 1:
            raise ValidationError("pair_window must be a positive integer")
        if type(config.seed) is not int or not 0 <= config.seed < 2**31:
            raise ValidationError("seed must be in [0, 2^31)")
        if config.geometry_scope != "all_selected_views_transductive":
            raise ValidationError(
                "Built-in pipeline supports only explicitly transductive SfM; use a new pipeline for train-only localization"
            )
        if not isinstance(config.plugins, (list, tuple)) or not all(
            isinstance(p, str) for p in config.plugins
        ):
            raise ValidationError("plugins must be a list of module names")
        return config

    @classmethod
    def load(cls, path: Path) -> PipelineConfig:
        return cls.parse(read_json(path))
