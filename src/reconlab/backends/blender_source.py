"""Optional Blender 4.2 source renderer. Run in its isolated bpy environment."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..io import content_hash, file_hash, read_json, write_json
from ..rendering import GifExporter
from ..runtime import require_version
from ..synthetic import SyntheticPolicy, write_input_evidence
from ..types import RenderArtifact


class BlenderObjectRenderer:
    def __init__(self, policy: SyntheticPolicy, threads: int = 8, device: str = "CPU"):
        if device not in {"CPU", "CUDA", "OPTIX"}:
            raise ValueError(f"Unsupported Cycles backend: {device}")
        self.policy = policy
        self.threads = threads
        self.device = device

    def setup(self, source: Path) -> dict:
        require_version("bpy", "4.2.0")
        import bpy
        from mathutils import Vector

        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(source), merge_vertices=True)
        scene = bpy.context.scene
        scene.frame_set(1)
        meshes = [obj for obj in scene.objects if obj.type == "MESH"]
        if not meshes:
            raise ValueError("Source contains no meshes")
        corners = np.array(
            [tuple(obj.matrix_world @ Vector(p)) for obj in meshes for p in obj.bound_box]
        )
        lower, upper = corners.min(0), corners.max(0)
        extent = float((upper - lower).max())
        if not np.isfinite(extent) or extent <= 0:
            raise ValueError("Source has invalid bounding box")
        center = (lower + upper) / 2
        roots = [obj for obj in scene.objects if obj.parent is None]
        from mathutils import Matrix

        transform = Matrix.Scale(1 / extent, 4) @ Matrix.Translation(Vector(-center))
        for obj in roots:
            obj.matrix_world = transform @ obj.matrix_world
        for obj in list(scene.objects):
            if obj.type in {"CAMERA", "LIGHT"}:
                bpy.data.objects.remove(obj, do_unlink=True)
        scene.render.engine = "CYCLES"
        devices = []
        if self.device == "CPU":
            scene.cycles.device = "CPU"
        else:
            preferences = bpy.context.preferences.addons["cycles"].preferences
            preferences.compute_device_type = self.device
            available = preferences.get_devices_for_type(self.device)
            selected = [d for d in available if d.type == self.device]
            if not selected:
                raise RuntimeError(f"No {self.device} Cycles device; refusing CPU fallback")
            selected_id = selected[0].id
            for device in preferences.devices:
                device.use = device.id == selected_id and device.type == self.device
            devices = [{"name": selected[0].name, "id": selected_id, "type": self.device}]
            scene.cycles.device = "GPU"
        scene.cycles.samples = self.policy.samples
        scene.cycles.use_denoising = True
        scene.cycles.seed = self.policy.seed
        scene.render.threads_mode = "FIXED"
        scene.render.threads = self.threads
        scene.render.resolution_x = self.policy.width
        scene.render.resolution_y = self.policy.height
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.film_transparent = True
        scene.view_settings.view_transform = "Standard"
        scene.world = bpy.data.worlds.new("World")
        scene.world.use_nodes = True
        scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
        scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.7
        for name, location, energy, size in [
            ("key", (3, -4, 5), 500, 4),
            ("fill", (-4, 1, 3), 300, 5),
        ]:
            data = bpy.data.lights.new(name, "AREA")
            data.energy, data.size = energy, size
            obj = bpy.data.objects.new(name, data)
            scene.collection.objects.link(obj)
            obj.location = location
            obj.rotation_euler = (-obj.location).to_track_quat("-Z", "Y").to_euler()
        camera_data = bpy.data.cameras.new("camera")
        camera_data.sensor_fit = "VERTICAL"
        camera_data.sensor_height = 32
        camera_data.lens = 32 / (2 * np.tan(np.deg2rad(self.policy.fov_deg) / 2))
        camera_data.clip_start, camera_data.clip_end = 0.01, 100
        camera = bpy.data.objects.new("camera", camera_data)
        scene.collection.objects.link(camera)
        scene.camera = camera
        return {
            "original_bounds": [lower.tolist(), upper.tolist()],
            "world_transform": [list(row) for row in transform],
            "blender_version": bpy.app.version_string,
            "cycles_backend": self.device,
            "cycles_devices": devices,
        }

    def render_frame(self, matrix: list, rgba: Path, rgb: Path) -> None:
        import bpy
        from mathutils import Matrix

        rgba.parent.mkdir(parents=True, exist_ok=True)
        rgb.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.camera.matrix_world = Matrix(np.asarray(matrix).reshape(4, 4).tolist())
        bpy.context.scene.render.filepath = str(rgba)
        bpy.ops.render.render(write_still=True)
        with Image.open(rgba) as image:
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            Image.alpha_composite(background, image.convert("RGBA")).convert("RGB").save(rgb)

    def render(self, source: Path, output: Path, metadata: dict, stage: str = "all") -> None:
        if stage not in {"all", "inputs", "reference"}:
            raise ValueError(stage)
        if (output / "COMPLETE.json").exists():
            raise FileExistsError(f"Already rendered: {output}")
        if stage in {"all", "inputs"} and (output / "INPUTS_COMPLETE.json").exists():
            raise FileExistsError(
                f"Inputs are immutable; use stage=reference or a new output: {output}"
            )
        output.mkdir(parents=True, exist_ok=True)
        normalized = self.setup(source)
        transforms = self.policy.transforms()
        _, orbit = self.policy.cameras()
        if stage in {"inputs", "all"}:
            write_json(output / "policy.json", asdict(self.policy))
            write_json(output / "transforms.json", transforms)
            write_json(output / "orbit.json", orbit)
            for frame in transforms["frames"]:
                name = Path(frame["file_path"]).name
                self.render_frame(
                    frame["transform_matrix"],
                    output / "source_rgba" / name,
                    output / frame["file_path"],
                )
            write_input_evidence(
                output,
                self.policy,
                {**metadata, **normalized, "glb_sha256": file_hash(source), "glb": str(source)},
            )
            write_json(
                output / "INPUTS_COMPLETE.json",
                {"inputs_sha256": file_hash(output / "inputs.json")},
            )
        else:
            evidence = read_json(output / "inputs.json")
            if content_hash(evidence["policy"]) != content_hash(asdict(self.policy)) or evidence[
                "source"
            ]["glb_sha256"] != file_hash(source):
                raise ValueError(
                    "Reference rendering must use the original source and input policy"
                )
            if content_hash(read_json(output / "orbit.json")) != content_hash(orbit):
                raise ValueError("Reference orbit changed")
        if stage == "inputs":
            return
        frames = []
        for index, camera in enumerate(orbit["camera_path"]):
            path = output / "reference_orbit" / f"frame_{index:05d}.png"
            self.render_frame(
                camera["camera_to_world"], output / "reference_rgba" / path.name, path
            )
            frames.append(path)
        GifExporter().export(
            RenderArtifact(
                tuple(frames),
                self.policy.fps,
                {"kind": "original_mesh_reference_NOT_reconstruction"},
            ),
            output / "source_mesh_reference.gif",
        )
        write_json(
            output / "COMPLETE.json",
            {
                "kind": "synthetic_input_render_complete",
                "images": len(transforms["frames"]),
                "orbit_frames": len(frames),
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("CPU", "CUDA", "OPTIX"), default="CPU")
    parser.add_argument("--stage", choices=("all", "inputs", "reference"), default="all")
    args = parser.parse_args()
    renderer = BlenderObjectRenderer(SyntheticPolicy(), args.threads, args.device)
    for item in read_json(args.catalog)["objects"]:
        output = args.output / item["id"]
        if not (output / "COMPLETE.json").exists():
            renderer.render(Path(item["local_path"]), output, item, args.stage)


if __name__ == "__main__":
    main()
    # bpy 4.2 in this headless environment can segfault during Python teardown,
    # after every output has closed and COMPLETE/INPUTS_COMPLETE is written.
    # Only a normally completed CLI reaches this exit. Exceptions remain errors.
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
