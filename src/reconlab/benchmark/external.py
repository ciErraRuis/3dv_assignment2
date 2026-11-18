"""Run native systems in their own dependency environments, with RGB-only inputs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .core import Geometry, ReconstructionMethod


class ExternalReconstruction(ReconstructionMethod):
    def __init__(self, name: str, config: dict):
        if name != "gluemap":
            raise ValueError(name)
        self.config = config

    def reconstruct(self, images: Path, output: Path) -> Geometry:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = (
            "/mnt/afs/260010098/data/envs/3dv-geometry-native/lib:"
            "/mnt/afs/260010098/data/envs/3dv-assignment-render/native_libs/usr/lib/x86_64-linux-gnu:"
            + env.get("LD_LIBRARY_PATH", "")
        )
        command = [
            self.config["gluemap_python"],
            "-m",
            "reconlab.benchmark.gluemap_worker",
            "--source",
            str(Path(self.config["sources"]) / "gluemap"),
            "--weights",
            str(Path(self.config["output"]) / "weights"),
            "--images",
            str(images),
            "--output",
            str(output),
        ]
        with (output / "native.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        return Geometry.load(output)
