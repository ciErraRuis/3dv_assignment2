"""Resumable experiment worker with per-object failure records and a live gallery."""

from __future__ import annotations

import argparse
import gc
import html
import json
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .core import save_json, sha256


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare_input(source: Path, destination: Path, expected_views=36):
    """Copy no data and expose no camera JSON to algorithms; verify each source hash."""
    records = json.loads((source / "inputs.json").read_text())
    data = json.loads((source / "transforms.json").read_text())
    if sha256(source / "transforms.json") != records["transforms_sha256"]:
        raise ValueError("Changed input provenance")
    hashes = {i["path"]: i["sha256"] for i in records["images"]}
    frames = [f for f in data["frames"] if f["split"] == "train"]
    if len(frames) != expected_views:
        raise ValueError("Invalid train split")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "images").mkdir(exist_ok=True)
    manifest = []
    for f in frames:
        p = source / f["file_path"]
        digest = sha256(p)
        if digest != hashes[f["file_path"]]:
            raise ValueError("Changed RGB: " + str(p))
        with Image.open(p) as im:
            if im.size != (512, 512):
                raise ValueError("Unexpected dimensions")
        link = destination / "images" / p.name
        if not link.exists():
            link.symlink_to(p)
        manifest.append({"name": p.name, "sha256": digest, "size": [512, 512]})
    if len(list((destination / "images").glob("*.png"))) != expected_views:
        raise ValueError("Held-out or unexpected images entered reconstruction folder")
    save_json(
        destination / "input_manifest.json",
        {
            "frames": manifest,
            "split": "train",
            "camera_metadata_supplied": False,
            "preprocessing": "full-frame 512 RGB",
        },
    )
    return destination / "images"


class Gallery:
    def __init__(self, config):
        self.config = config
        self.root = Path(config["output"])

    def build(self):
        rows, records = [], []
        for obj in self.config["objects"]:
            for method in self.config["methods"]:
                directory = self.root / "results" / method / obj["uid"]
                p = directory / "status.json"
                status = json.loads(p.read_text()) if p.exists() else {"state": "PENDING"}
                record = {"object": obj["name"], "uid": obj["uid"], "method": method, **status}
                records.append(record)
                prefix = f"results/{method}/{obj['uid']}/"
                visuals = sorted((directory / "visuals").glob("*.png"))[:1]
                match = (
                    f'<a href="{prefix}visuals/{visuals[0].name}"><img loading="lazy" src="{prefix}visuals/{visuals[0].name}" width="400"></a>'
                    if visuals
                    else ""
                )
                gif = directory / "gaussian/reconstruction.gif"
                render = (
                    f'<img loading="lazy" src="{prefix}gaussian/reconstruction.gif" width="240">'
                    if gif.exists()
                    else ""
                )
                detail = html.escape(status.get("error", status.get("stage", "")))
                rows.append(
                    f"<tr><td>{html.escape(obj['name'])}<br>{method}</td><td>{status['state']}<br>{detail}</td><td>{match}</td><td>{render}</td></tr>"
                )
        page = '<!doctype html><meta charset="utf-8"><title>10 objects: matching and reconstruction</title>'
        page += "<style>body{font:15px system-ui;margin:24px;background:#f6f7f9}td,th{border:1px solid #ddd;padding:12px;vertical-align:top}table{border-collapse:collapse;width:100%}img{max-width:100%}</style>"
        page += f"<h1>10 objects × matching / reconstruction methods</h1><p>{self.config.get('input_views', 36)} train RGB views; estimated cameras; source 512×512. Validation/test RGB excluded from reconstruction. Internal full-frame resizing disclosed per method. Pending and failed runs remain visible.</p>"
        page += "<p>These public Objaverse candidates are engineering experiments, not a claim to be Assignment 1 cleaned data. Match images show actual correspondences; GIFs are Gaussian reconstructions, not source-mesh renders.</p>"
        page += (
            f"<p>Updated {now()}</p><table><tr><th>Object / method</th><th>Status</th><th>Point matching</th><th>Reconstruction GIF</th></tr>"
            + "".join(rows)
            + "</table>"
        )
        temporary = self.root / ("report." + uuid.uuid4().hex + ".html.tmp")
        temporary.write_text(page)
        temporary.replace(self.root / "report.html")
        save_json(self.root / "summary.json", {"updated_at": now(), "runs": records})


class BenchmarkRunner:
    def __init__(self, config: dict):
        self.config, self.root = config, Path(config["output"])
        self.gallery = Gallery(config)

    def run_method(self, name: str):
        import torch

        from .gaussian import GaussianBenchmark
        from .matchers import create_matcher
        from .native import Mast3rSfm
        from .sfm import PairwiseReconstruction

        torch.set_num_threads(4)
        torch.manual_seed(42)
        sources, weights = Path(self.config["sources"]), self.root / "weights"
        if not torch.cuda.is_available() and name != "sift":
            raise RuntimeError("No CUDA device")
        method = None
        load_error = None
        try:
            if name == "mast3r_sfm":
                method = Mast3rSfm(sources, weights)
            elif name == "mast3r_slam":
                from .slam import Mast3rSlam

                method = Mast3rSlam(self.config)
            elif name == "mvroma":
                from .multiview import MultiViewRoma

                method = MultiViewRoma(sources, weights)
            elif name == "gluemap":
                from .external import ExternalReconstruction

                method = ExternalReconstruction(name, self.config)
            else:
                if self.config.get("pair_cache_root"):
                    from .sparse import CachedPairMatcher

                    matcher = CachedPairMatcher(name, self.config, sources, weights)
                else:
                    matcher = create_matcher(name, sources, weights)
                method = PairwiseReconstruction(matcher, name)
        except Exception:
            load_error = traceback.format_exc()
        for obj in self.config["objects"]:
            destination = self.root / "results" / name / obj["uid"]
            destination.mkdir(parents=True, exist_ok=True)
            status_path = destination / "status.json"
            if status_path.exists() and json.loads(status_path.read_text())["state"] == "COMPLETE":
                continue
            if load_error:
                (destination / "error.log").write_text(load_error)
                save_json(
                    status_path,
                    {
                        "state": "SETUP_FAILED",
                        "error": load_error.splitlines()[-1],
                        "checked_at": now(),
                    },
                )
                self.gallery.build()
                continue
            source = Path(self.config["source_inputs"]) / obj["uid"]
            start = time.monotonic()
            stage = "waiting_for_source"
            while not (source / "inputs.json").exists():
                save_json(
                    status_path, {"state": "WAITING_INPUT", "stage": stage, "checked_at": now()}
                )
                self.gallery.build()
                if time.monotonic() - start > 7200:
                    raise TimeoutError("Source producer did not finish")
                time.sleep(30)
            try:
                image_dir = prepare_input(
                    source, self.root / "inputs" / obj["uid"], self.config.get("input_views", 36)
                )
                stage = "matching_and_geometry"
                save_json(status_path, {"state": "RUNNING", "stage": stage, "checked_at": now()})
                self.gallery.build()
                geometry = method.reconstruct(image_dir, destination)
                geometry.save(destination)
                stage = "gaussian_training"
                save_json(
                    status_path,
                    {
                        "state": "RUNNING",
                        "stage": stage,
                        "registered": len(geometry.names),
                        "checked_at": now(),
                    },
                )
                self.gallery.build()
                gaussian = GaussianBenchmark(self.config["gaussian_steps"])
                params, center, scale = gaussian.fit(geometry, image_dir, destination / "gaussian")
                stage = "heldout_evaluation_and_gif"
                save_json(status_path, {"state": "RUNNING", "stage": stage, "checked_at": now()})
                gaussian.evaluate(geometry, source, destination / "gaussian", params, center, scale)
                del params
                save_json(
                    status_path,
                    {
                        "state": "COMPLETE",
                        "registered": len(geometry.names),
                        "input_views": self.config.get("input_views", 36),
                        "seconds": time.monotonic() - start,
                        "checked_at": now(),
                    },
                )
            except Exception as error:
                (destination / "error.log").write_text(traceback.format_exc())
                save_json(
                    status_path,
                    {
                        "state": "FAILED",
                        "stage": stage,
                        "error": f"{type(error).__name__}: {error}",
                        "seconds": time.monotonic() - start,
                        "checked_at": now(),
                    },
                )
            self.gallery.build()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.method not in config["methods"]:
        raise ValueError("Method outside frozen experiment")
    BenchmarkRunner(config).run_method(args.method)


if __name__ == "__main__":
    main()
