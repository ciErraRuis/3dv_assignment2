"""Package actual ten-object artifacts and source code without pretrained weights."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .core import save_json, sha256


class SubmissionExporter:
    def __init__(self, config, repository: Path):
        self.config, self.repository = config, repository
        self.root = Path(config["output"])

    def export(self, destination: Path, method: str):
        if destination.exists():
            raise FileExistsError(
                "Use a new destination; an existing submission is never overwritten"
            )
        sources = []
        for obj in self.config["objects"]:
            folder = self.root / "results" / method / obj["uid"]
            status = json.loads((folder / "status.json").read_text())
            required = [
                "matching.json",
                "geometry.npz",
                "geometry.json",
                "gaussian/reconstruction.gif",
                "gaussian/training.json",
                "gaussian/render.json",
                "gaussian/render_metrics.json",
            ]
            if status["state"] != "COMPLETE" or any(not (folder / f).is_file() for f in required):
                raise ValueError(
                    f"Incomplete object {obj['uid']}; no partial submission is packaged"
                )
            if len(list((folder / "gaussian/orbit").glob("*.png"))) != 72:
                raise ValueError(f"Missing authoritative orbit frames: {obj['uid']}")
            sources.append((obj, folder))
        shutil.copytree(
            self.repository,
            destination / "repository",
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "*.pyc"
            ),
        )
        records = []
        for obj, folder in sources:
            target = destination / "repository/results" / obj["uid"]
            target.mkdir(parents=True)
            for name in ["matches", "visuals", "model_inputs"]:
                if (folder / name).exists():
                    shutil.copytree(folder / name, target / name)
            for name in [
                "matching.json",
                "geometry.npz",
                "geometry.json",
                "geometry.html",
                "pose_metrics.json",
                "status.json",
            ]:
                if (folder / name).exists():
                    shutil.copy2(folder / name, target / name)
            for name in [
                "reconstruction.gif",
                "training.json",
                "render.json",
                "render_metrics.json",
                "pose_metrics.json",
            ]:
                shutil.copy2(folder / "gaussian" / name, target / ("gaussian_" + name))
            records.append(
                {
                    "uid": obj["uid"],
                    "name": obj["name"],
                    "method": method,
                    "source": str(folder),
                    "camera_condition": "estimated from train RGB",
                    "quality": "Execution complete; consult metrics and visual results. Not an automatic quality endorsement.",
                }
            )
        manifest = {
            str(p.relative_to(destination)): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(destination.rglob("*"))
            if p.is_file()
        }
        save_json(
            destination / "artifact_manifest.json",
            {
                "files": manifest,
                "objects": records,
                "group_number": "[待填写]",
                "members_and_contributions": "[待填写]",
                "github_url": "[待填写]",
                "dataset": "fixed ten public Objaverse samples, explicitly authorized by user; not claimed Assignment1-cleaned data",
                "omitted_large_artifacts": "pretrained models, native build caches, Gaussian checkpoints and 72-frame PNG sequences stay in original data directory with recorded hashes/paths; matches and GIFs included",
                "publication": "local package only; no GitHub upload performed",
            },
        )
        return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    SubmissionExporter(json.loads(args.config.read_text()), args.repository).export(
        args.destination, args.method
    )


if __name__ == "__main__":
    main()
