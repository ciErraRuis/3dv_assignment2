"""Isolated method batches and selected full-budget Gaussian runs."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

from .core import Geometry, save_json, sha256


class SparseWorker:
    def __init__(self, snapshot, case):
        self.snapshot, self.case = snapshot, case
        self.plan = json.loads((snapshot / "plan.json").read_text())
        self.config = json.loads((snapshot / "matching.json").read_text())
        self.config["sources"] = self.plan["method_sources"].get(
            case["method"], self.config["sources"]
        )
        self.root = Path(self.config["output"])
        self.gt = Path(self.plan["gt_root"])

    def load_native(self):
        import torch

        assert torch.cuda.is_available()
        pin = self.plan["native_binary"]
        assert sha256(Path(pin["path"])) == pin["sha256"]
        spec = importlib.util.spec_from_file_location("gsplat_cuda", pin["path"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["gsplat.csrc"] = module
        import gsplat

        gsplat.csrc = module
        from gsplat.cuda._backend import _C

        assert _C is module

    def gaussian(self):
        import torch

        from .estimated_scene import EstimatedScene
        from .gt_rendering import GaussianComparison, GroundTruthScene

        self.load_native()
        kind, method = self.case["kind"], self.case["method"]
        final = kind != "gt_screen"
        objects = self.config["objects"]
        if self.case.get("uid"):
            objects = [o for o in objects if o["uid"] == self.case["uid"]]
        for obj in objects:
            uid = obj["uid"]
            source = Path(self.config["source_inputs"]) / uid
            output = (
                self.gt
                / (
                    "final/estimated"
                    if kind == "estimated_final"
                    else "final/gt"
                    if final
                    else "results"
                )
                / method
                / uid
            )
            if (output / "status.json").exists():
                continue  # A completed attempt, including an honest failure, is not silently retried.
            save_json(
                output / "status.json",
                {"state": "RUNNING", "checked_at": time.time(), "kind": kind},
            )
            started = time.monotonic()
            try:
                if kind == "estimated_final":
                    frontend = self.case["frontend"]
                    seed_path = self.root / "results" / frontend / uid
                    seed = Geometry.load(seed_path)
                    scene = EstimatedScene(source, seed)
                else:
                    seed_path = self.gt / "initialization" / uid
                    seed = Geometry.load(seed_path)
                    scene = GroundTruthScene(source, seed)
                    scene.initialization_description = "Common LoMa correspondences from exactly six selected RGB images triangulated with their GT cameras; no GT mesh or 36-view geometry"
                save_json(
                    output / "initialization.json",
                    {
                        "geometry": str(seed_path / "geometry.npz"),
                        "geometry_sha256": sha256(seed_path / "geometry.npz"),
                        "source_inputs_sha256": sha256(source / "inputs.json"),
                        "training_names": seed.names,
                        "seed": 42,
                    },
                )
                GaussianComparison(method, 30000 if final else 3000, scale_schedule=True).fit(
                    scene, output
                )
                status = {"state": "COMPLETE"}
            except Exception as error:
                (output / "error.log").write_text(traceback.format_exc())
                status = {"state": "FAILED", "error": str(error)}
            save_json(
                output / "status.json",
                {
                    **status,
                    "checked_at": time.time(),
                    "seconds": time.monotonic() - started,
                    "kind": kind,
                },
            )
            gc.collect()
            torch.cuda.empty_cache()

    def run(self):
        import torch

        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        torch.set_num_threads(4)
        if self.case["kind"] == "matching":
            from .runner import BenchmarkRunner
            from .visuals import export_geometry

            runner = BenchmarkRunner(self.config)
            runner.gallery.build = lambda: None
            runner.run_method(self.case["method"])
            for obj in self.config["objects"]:
                output = self.root / "results" / self.case["method"] / obj["uid"]
                if (output / "geometry.npz").exists():
                    try:
                        export_geometry(
                            Geometry.load(output),
                            Path(self.config["source_inputs"]) / obj["uid"],
                            output,
                        )
                    except Exception:
                        (output / "visualization_error.log").write_text(traceback.format_exc())
        elif self.case["kind"] == "views":
            from .sparse import SparseGeometryStudy

            SparseGeometryStudy(self.config).run(self.case["method"])
        else:
            self.gaussian()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    SparseWorker(args.snapshot, json.loads(args.case)).run()


if __name__ == "__main__":
    main()
