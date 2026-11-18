"""Run a frozen GPU-rendering/3DGS batch with isolated per-object failures."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from .batch_analysis import BatchGallery, ObjectAnalyzer
from .io import file_hash, read_json, utc_now, write_json
from .synthetic import PosedImageDataset, SyntheticPolicy


class InputAuditor:
    def audit(self, inputs: Path, item: dict) -> dict:
        dataset = PosedImageDataset(inputs)
        if dataset.provenance["source"]["glb_sha256"] != item["audit"]["sha256"]:
            raise ValueError("Source GLB differs from frozen catalog")
        if dataset.provenance["source"].get("cycles_backend") != "CUDA":
            raise ValueError("Batch requires explicit CUDA source rendering")
        splits = {}
        for split, count in [("train", 36), ("val", 6), ("test", 6)]:
            images, _ = dataset.load_split(split)
            if images.shape != (count, 512, 512, 3):
                raise ValueError(f"Unexpected {split} input dimensions")
            splits[split] = list(images.shape)
        bounds, empty = [], []
        for frame in dataset.frames:
            name = Path(frame["file_path"]).name
            with Image.open(inputs / "source_rgba" / name) as image:
                mask = np.array(image.getchannel("A")) > 0
            yy, xx = np.nonzero(mask)
            if not len(xx):
                empty.append(name)
            elif xx.min() == 0 or yy.min() == 0 or xx.max() == 511 or yy.max() == 511:
                raise ValueError(f"Object touches image boundary: {name}; no automatic crop/zoom")
            bounds.append({"view": name, "foreground_pixels": int(mask.sum())})
        return {
            "checked_at": utc_now(),
            "split_shapes": splits,
            "bounds": bounds,
            "empty_views": empty,
            "full_frame": True,
            "input_sha256": file_hash(inputs / "inputs.json"),
        }


class CollectionBatchRunner:
    def __init__(self, config: Path):
        self.config_path = config.resolve()
        self.config = read_json(config)
        self.root = Path(self.config["output"])
        self.task = Path(self.config["task"])
        self.catalog_path = Path(self.config["catalog"])
        self.items = read_json(self.catalog_path)["objects"]
        self.snapshot = Path(__file__).resolve().parents[2]
        self.env = os.environ.copy()
        self.env.update(
            PYTHONPATH=str(self.snapshot / "src"),
            OMP_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="1",
            PYTHONUNBUFFERED="1",
        )
        self.last_heartbeat = 0.0

    def record(self, uid: str, phase: str, state: str, **extra) -> None:
        write_json(
            self.root / "progress" / uid / f"{phase}.json",
            {"uid": uid, "state": state, "checked_at": utc_now(), **extra},
        )

    def heartbeat(self, uid: str | None, phase: str, force: bool = False) -> None:
        if not force and time.monotonic() - self.last_heartbeat < 15:
            return
        completed = sum(
            (self.root / "results" / x["id"] / "COMPLETE.json").exists() for x in self.items
        )
        sources = sum(
            (self.root / "inputs" / x["id"] / "COMPLETE.json").exists() for x in self.items
        )
        step = 0
        if uid:
            progress = self.root / "results" / uid / "training/progress.jsonl"
            if progress.exists():
                for line in progress.read_text().splitlines():
                    try:
                        step = json.loads(line)["optimizer_step"]
                    except (ValueError, KeyError):
                        pass
        status = {
            "state": "RUNNING" if completed or step >= 100 else "STARTING",
            "phase": phase,
            "checked_at": utc_now(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "current_uid": uid,
            "current_optimizer_step": step,
            "source_objects_complete": sources,
            "reconstructions_complete": completed,
            "total": len(self.items),
            "report": str(self.root / "report.html"),
        }
        attempt = os.environ.get("RECONLAB_ATTEMPT", "0001")
        receipt = self.task / f"execution/attempts/{attempt}/receipt.json"
        if receipt.exists():
            status["job_id"] = read_json(receipt)["job_id"]
        write_json(self.root / "HEARTBEAT.json", status)
        write_json(self.task / "execution/status.json", status)
        self.last_heartbeat = time.monotonic()

    def command(
        self, args: list[str], log: Path, uid: str | None, phase: str, timeout: int
    ) -> float:
        start = time.monotonic()
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            stream.write(json.dumps({"started_at": utc_now(), "argv": args}) + "\n")
            stream.flush()
            process = subprocess.Popen(args, env=self.env, stdout=stream, stderr=subprocess.STDOUT)
            try:
                while process.poll() is None:
                    self.heartbeat(uid, phase)
                    if (self.task / "CANCEL.json").exists():
                        raise RuntimeError("User cancellation marker detected")
                    if time.monotonic() - start > timeout:
                        raise TimeoutError(f"{phase} exceeded fixed {timeout}s budget")
                    time.sleep(2)
            except BaseException:
                process.kill()
                process.wait()
                raise
        if process.returncode:
            raise RuntimeError(f"{phase} exited {process.returncode}; log: {log}")
        return time.monotonic() - start

    def source(self, item: dict) -> None:
        uid = item["id"]
        output = self.root / "inputs" / uid
        if file_hash(Path(item["local_path"])) != item["audit"]["sha256"]:
            raise ValueError("Raw object hash changed")
        if not (output / "COMPLETE.json").exists():
            if output.exists():
                raise RuntimeError("Partial source output exists; inspect before recovery")
            one = self.root / "catalogs" / f"{uid}.json"
            write_json(one, {"objects": [item]})
            seconds = self.command(
                [
                    self.config["render_python"],
                    "-m",
                    "reconlab.backends.blender_source",
                    "--catalog",
                    str(one),
                    "--output",
                    str(self.root / "inputs"),
                    "--threads",
                    "2",
                    "--device",
                    "CUDA",
                    "--stage",
                    "all",
                ],
                self.root / "logs" / uid / "source.log",
                uid,
                "GPU_SOURCE_RENDERING",
                1800,
            )
        else:
            seconds = None
        audit = InputAuditor().audit(output, item)
        write_json(self.root / "progress" / uid / "input_audit.json", audit)
        self.record(
            uid,
            "source",
            "COMPLETE",
            seconds=seconds,
            input_sha256=audit["input_sha256"],
            empty_views=audit["empty_views"],
        )

    def train(self, item: dict) -> None:
        uid = item["id"]
        result = self.root / "results" / uid
        if (result / "COMPLETE.json").exists():
            self.record(uid, "gpu", "COMPLETE", resumed_existing_complete_output=True)
            return
        if result.exists():
            raise RuntimeError("Partial reconstruction output exists; inspect before recovery")
        self.record(uid, "gpu", "STARTING")
        seconds = self.command(
            [
                self.config["gpu_python"],
                "-m",
                "reconlab.backends.gsplat_known",
                "--inputs",
                str(self.root / "inputs"),
                "--output",
                str(self.root / "results"),
                "--config",
                str(self.snapshot / "src/reconlab/defaults/gaussians.json"),
                "--objects",
                uid,
            ],
            self.root / "logs" / uid / "training.log",
            uid,
            "3DGS_TRAINING",
            2400,
        )
        self.record(
            uid,
            "gpu",
            "COMPLETE",
            seconds=seconds,
            checkpoint=str(result / "training/step_003000.pt"),
        )

    def diagnose(self, item: dict) -> None:
        import cv2

        from .pilot_report import PilotMatching

        cv2.setNumThreads(1)
        uid = item["id"]
        try:
            self.record(uid, "matching", "RUNNING")
            summary = self.root / "diagnostics" / uid / "matching/summary.json"
            if not summary.exists():
                PilotMatching().run(self.root / "inputs", self.root / "diagnostics", [uid])
            self.record(uid, "matching", "COMPLETE")
        except Exception as exc:
            self.record(uid, "matching", "FAILED", error=str(exc), traceback=traceback.format_exc())
        try:
            if (self.root / "results" / uid / "COMPLETE.json").exists():
                ObjectAnalyzer().analyze(self.root, uid)
                self.record(uid, "analysis", "COMPLETE")
        except Exception as exc:
            self.record(uid, "analysis", "FAILED", error=str(exc), traceback=traceback.format_exc())

    def run(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / "batch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.heartbeat(None, "GPU_PREFLIGHT", True)
            self.command(
                [
                    self.config["render_python"],
                    "-m",
                    "reconlab.batch100",
                    "probe",
                    "--config",
                    str(self.config_path),
                ],
                self.root / "logs/gpu_probe.log",
                None,
                "GPU_PREFLIGHT",
                1200,
            )
            gallery = BatchGallery()
            gallery.build(self.root, self.catalog_path)
            with ThreadPoolExecutor(max_workers=1) as pool:
                futures = []
                for item in self.items:
                    uid = item["id"]
                    if (self.task / "CANCEL.json").exists():
                        raise RuntimeError("User cancellation marker detected")
                    self.record(uid, "source", "RUNNING")
                    try:
                        self.source(item)
                    except Exception as exc:
                        self.record(
                            uid,
                            "source",
                            "FAILED",
                            error=str(exc),
                            traceback=traceback.format_exc(),
                        )
                        gallery.build(self.root, self.catalog_path)
                        continue
                    try:
                        self.train(item)
                    except Exception as exc:
                        self.record(
                            uid, "gpu", "FAILED", error=str(exc), traceback=traceback.format_exc()
                        )
                    futures.append(pool.submit(self.diagnose, item))
                    gallery.build(self.root, self.catalog_path)
                    self.heartbeat(uid, "NEXT_OBJECT", True)
                for future in futures:
                    future.result()
            gallery.build(self.root, self.catalog_path)
            summary = read_json(self.root / "batch_summary.json")["summary"]
            state = {
                "state": "EXECUTION_COMPLETE",
                "checked_at": utc_now(),
                "summary": summary,
                "report": str(self.root / "report.html"),
                "note": "All objects attempted; completed artifacts and failures are reported separately.",
            }
            write_json(self.root / "COMPLETE.json", state)
            write_json(self.task / "execution/status.json", state)
            write_json(self.task / "evaluation/result.json", state)


def gpu_probe(config: Path) -> None:
    from .backends.blender_source import BlenderObjectRenderer

    settings = read_json(config)
    root = Path(settings["output"]) / "validation/gpu_probe"
    item = next(
        x
        for x in read_json(Path(settings["catalog"]))["objects"]
        if x["uid"] == "3bb4f120d0df42fcba3dc4f550aaa9e5"
    )
    renderer = BlenderObjectRenderer(SyntheticPolicy(), threads=2, device="CUDA")
    metadata = renderer.setup(Path(item["local_path"]))
    matrix = SyntheticPolicy().transforms()["frames"][0]["transform_matrix"]
    renderer.render_frame(matrix, root / "rgba.png", root / "rgb.png")
    with Image.open(root / "rgb.png") as image:
        pixels = np.array(image)
        if image.size != (512, 512) or image.mode != "RGB":
            raise ValueError("GPU probe violates input specification")
    old = Path(settings["old_pilot"]) / "inputs/fire_extinguisher/images/view_000.png"
    with Image.open(old) as image:
        previous = np.array(image)
    error = (pixels.astype(np.float64) - previous) / 255
    write_json(
        root / "result.json",
        {
            "checked_at": utc_now(),
            "device": metadata,
            "rgb_sha256": file_hash(root / "rgb.png"),
            "previous_cpu_input": str(old),
            "cpu_gpu_mae": float(np.abs(error).mean()),
            "cpu_gpu_psnr_db": -10 * np.log10(max(float((error**2).mean()), 1e-12)),
            "purpose": "GPU backend smoke test and same-view CPU/GPU image difference; "
            "not a reconstruction quality benchmark. All batch inputs are regenerated on CUDA.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "probe", "report"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "probe":
        gpu_probe(args.config)
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    elif args.action == "report":
        settings = read_json(args.config)
        print(BatchGallery().build(Path(settings["output"]), Path(settings["catalog"])))
    else:
        CollectionBatchRunner(args.config).run()


if __name__ == "__main__":
    main()
