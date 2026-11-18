"""Small CLI over the class API. No default GPU work or model downloads."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from .config import PipelineConfig
from .errors import ReconlabError
from .io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reconlab")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "doctor", help="Inspect optional dependencies without importing GPU libraries"
    )
    check = commands.add_parser("check", help="Validate manifest schema and file existence")
    check.add_argument("--manifest", type=Path, required=True)
    plan = commands.add_parser("plan", help="Expand one experiment suite; no execution")
    plan.add_argument("--suite", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Execute one variant across all manifest objects")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--until", choices=["prepare", "matching", "sfm", "train"], default="matching")
    run.add_argument("--continue-on-error", action="store_true")
    report = commands.add_parser("report")
    report.add_argument("--run", type=Path, required=True)
    train = commands.add_parser(
        "train", help="Train from exact existing input/SfM artifacts, without recomputing poses"
    )
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--prepared", type=Path, required=True)
    train.add_argument("--sfm", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    orbit = commands.add_parser("orbit", help="Create an explicit OpenGL camera path")
    orbit.add_argument("--center", type=float, nargs=3, required=True)
    orbit.add_argument("--radius", type=float, required=True)
    orbit.add_argument("--elevation", type=float, default=15)
    orbit.add_argument("--frames", type=int, default=120)
    orbit.add_argument("--fps", type=float, default=24)
    orbit.add_argument("--width", type=int, default=640)
    orbit.add_argument("--height", type=int, default=480)
    orbit.add_argument("--output", type=Path, required=True)
    render = commands.add_parser("render", help="Render an exact Nerfstudio model checkpoint")
    render.add_argument("--model", type=Path, required=True)
    render.add_argument("--camera-path", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    gif = commands.add_parser("gif", help="Encode a render.json PNG sequence as GIF")
    gif.add_argument("--render", type=Path, required=True)
    gif.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate explicitly paired test renders")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            print(
                json.dumps(
                    {
                        name: importlib.util.find_spec(name) is not None
                        for name in (
                            "numpy",
                            "PIL",
                            "cv2",
                            "h5py",
                            "torch",
                            "lightglue",
                            "hloc",
                            "pycolmap",
                            "nerfstudio",
                        )
                    },
                    indent=2,
                )
            )
        elif args.command == "check":
            from .data import JsonDatasetLoader

            objects = JsonDatasetLoader().load(args.manifest.resolve())
            print(json.dumps({obj.id: len(obj.views) for obj in objects}, indent=2))
        elif args.command == "plan":
            from .experiments import ExperimentSuite

            ExperimentSuite.load(args.suite).write_plan(args.output.resolve())
            print(args.output.resolve())
        elif args.command == "run":
            from .pipeline import ReconstructionPipeline

            record = ReconstructionPipeline(PipelineConfig.load(args.config)).run(
                args.manifest.resolve(), args.output.resolve(), args.until, args.continue_on_error
            )
            print(
                json.dumps(
                    {
                        "status": record["status"],
                        "report": str(args.output.resolve() / "report.html"),
                    }
                )
            )
            return 0 if record["status"] == "STAGE_COMPLETE" else 1
        elif args.command == "report":
            from .reporting import HtmlReport

            print(HtmlReport().write(args.run.resolve()))
        elif args.command == "train":
            from .artifacts import ArtifactStore
            from .pipeline import ReconstructionPipeline

            config = PipelineConfig.load(args.config)
            pipeline = ReconstructionPipeline(config)
            store = ArtifactStore()
            model = pipeline.reconstructor.fit(
                store.load_prepared(args.prepared), store.load_sfm(args.sfm), args.output.resolve()
            )
            write_json(args.output.resolve() / "model.json", model)
        elif args.command == "orbit":
            from .rendering import OrbitCameraPath

            write_json(
                args.output.resolve(),
                OrbitCameraPath().build(
                    tuple(args.center),
                    args.radius,
                    args.elevation,
                    args.frames,
                    args.fps,
                    args.width,
                    args.height,
                ),
            )
        elif args.command == "render":
            from .backends.nerfstudio import NerfstudioRenderer, load_model

            NerfstudioRenderer().render(
                load_model(args.model), args.camera_path.resolve(), args.output.resolve()
            )
        elif args.command == "gif":
            from .io import read_json
            from .rendering import GifExporter
            from .types import RenderArtifact

            raw = read_json(args.render)
            artifact = RenderArtifact(
                tuple((args.render.parent / p).resolve() for p in raw["frames"]),
                raw["fps"],
                raw.get("metadata", {}),
            )
            print(GifExporter().export(artifact, args.output.resolve()))
        elif args.command == "evaluate":
            from .evaluation import ImageEvaluator

            ImageEvaluator().evaluate(args.manifest.resolve(), args.output.resolve())
    except (ReconlabError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
