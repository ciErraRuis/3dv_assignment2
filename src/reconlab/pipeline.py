"""Stage orchestration with dependency injection and explicit failure accounting."""

from __future__ import annotations

import importlib.metadata
import platform
import time
from pathlib import Path

from .config import PipelineConfig
from .data import JsonDatasetLoader
from .errors import ValidationError
from .factory import default_registry
from .frontend import MatchingFrontend
from .interfaces import (
    FeatureExtractor,
    FeatureMatcher,
    GeometricVerifier,
    PoseEstimator,
    Preprocessor,
    Reconstructor,
    ViewSelector,
)
from .io import content_hash, file_hash, utc_now, write_json
from .registry import ComponentRegistry
from .reporting import HtmlReport
from .selection import PairGenerator
from .types import ObjectViews


class ReconstructionPipeline:
    def __init__(self, config: PipelineConfig, registry: ComponentRegistry | None = None) -> None:
        self.config, self.registry = config, registry or default_registry()
        for plugin in config.plugins:
            self.registry.load_plugin(plugin)
        self._components = {}

    @property
    def reconstructor(self) -> Reconstructor:
        """Access the configured backend when reusing existing data/pose artifacts."""
        return self._component(Reconstructor, "reconstructor")

    def _component(self, interface, attribute):
        if attribute not in self._components:
            spec = getattr(self.config, attribute)
            options = dict(spec.options)
            if attribute == "verifier" and spec.name == "fundamental":
                options.setdefault("seed", self.config.seed)
            if attribute == "reconstructor" and spec.name in {"splatfacto", "nerfacto"}:
                options.setdefault("seed", self.config.seed)
            self._components[attribute] = self.registry.create(interface, spec.name, **options)
        return self._components[attribute]

    def run(
        self, manifest: Path, output: Path, until: str = "matching", continue_on_error: bool = False
    ) -> dict:
        if until not in {"prepare", "matching", "sfm", "train"}:
            raise ValidationError(f"Unknown stopping stage: {until}")
        objects = JsonDatasetLoader().load(manifest)
        output = output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", self.config)
        versions = {"python": platform.python_version()}
        for name in (
            "numpy",
            "Pillow",
            "opencv-python-headless",
            "h5py",
            "torch",
            "nerfstudio",
            "pycolmap",
            "lightglue",
        ):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        source_root = Path(__file__).parent
        write_json(
            output / "environment.json",
            {
                "versions": versions,
                "source_hashes": {
                    str(p.relative_to(source_root)): file_hash(p)
                    for p in sorted(source_root.rglob("*.py"))
                },
            },
        )
        record = {
            "experiment": self.config.name,
            "config_sha256": content_hash(self.config),
            "manifest": str(manifest.resolve()),
            "manifest_sha256": file_hash(manifest),
            "started_at": utc_now(),
            "status": "RUNNING",
            "until": until,
            "objects": [],
        }
        write_json(output / "run.json", record)
        interrupted = False
        try:
            for obj in objects:
                row = {"object_id": obj.id, "status": "RUNNING", "stages": {}}
                record["objects"].append(row)
                write_json(output / "run.json", record)
                try:
                    self._run_object(obj, output / obj.id, until, row)
                except Exception as exc:
                    row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
                    if not continue_on_error:
                        break
                finally:
                    write_json(output / "run.json", record)
        except BaseException:
            interrupted = True
            raise
        finally:
            attempted = {r["object_id"] for r in record["objects"]}
            for obj in objects:
                if obj.id not in attempted:
                    record["objects"].append({"object_id": obj.id, "status": "NOT_RUN"})
            if interrupted:
                for row in record["objects"]:
                    if row["status"] == "RUNNING":
                        row["status"] = "INTERRUPTED"
            record["status"] = (
                "INTERRUPTED"
                if interrupted
                else (
                    "FAILED"
                    if any(r["status"] in {"FAILED", "NOT_RUN"} for r in record["objects"])
                    else "STAGE_COMPLETE"
                )
            )
            record["finished_at"] = utc_now()
            write_json(output / "run.json", record)
            HtmlReport().write(output)
        return record

    def _run_object(self, obj: ObjectViews, output: Path, until: str, row: dict) -> None:
        selector = self._component(ViewSelector, "selector")
        selected = selector.select(obj.views)
        original = {view.id: view for view in obj.views}
        if len({view.id for view in selected}) != len(selected) or any(
            original.get(view.id) != view for view in selected
        ):
            raise ValidationError("Selector must return unique, unchanged source views")
        if not any(view.split == "train" for view in selected):
            raise ValidationError("Selector removed all training views")
        held_out = {v.id for v in obj.views if v.split != "train"}
        if {v.id for v in selected if v.split != "train"} != held_out:
            raise ValidationError("Selector changed the held-out view set")
        write_json(
            output / "selection.json",
            {"source_ids": [v.id for v in obj.views], "selected_ids": [v.id for v in selected]},
        )
        start = time.monotonic()
        prepared = self._component(Preprocessor, "preprocessor").prepare(
            ObjectViews(obj.id, selected), output / "data"
        )
        row["stages"]["prepare_seconds"] = time.monotonic() - start
        row["status"] = "PREPARED"
        if until == "prepare":
            return
        frontend = MatchingFrontend(
            self._component(FeatureExtractor, "extractor"),
            self._component(FeatureMatcher, "matcher"),
            self._component(GeometricVerifier, "verifier"),
            PairGenerator(self.config.pairing, self.config.pair_window),
        )
        start = time.monotonic()
        matches = frontend.run(prepared, output / "matching")
        row["stages"]["matching_seconds"] = time.monotonic() - start
        row["status"] = "MATCHED"
        if until == "matching":
            return
        start = time.monotonic()
        sfm = self._component(PoseEstimator, "pose_estimator").estimate(
            prepared, matches, output / "sfm"
        )
        write_json(output / "sfm" / "artifact.json", sfm)
        row["stages"]["sfm_seconds"] = time.monotonic() - start
        row["status"] = "POSES_ESTIMATED"
        if until == "sfm":
            return
        start = time.monotonic()
        model = self._component(Reconstructor, "reconstructor").fit(
            prepared, sfm, output / "training"
        )
        write_json(output / "training" / "model.json", model)
        row["stages"]["training_seconds"] = time.monotonic() - start
        row["status"] = "TRAINED"
