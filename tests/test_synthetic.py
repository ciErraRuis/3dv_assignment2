"""Coordinate and split integrity checks; these run without Blender or CUDA."""

import numpy as np
import pytest
from PIL import Image

from reconlab.backends.blender_source import BlenderObjectRenderer
from reconlab.backends.gsplat_known import opengl_to_view
from reconlab.errors import ValidationError
from reconlab.io import write_json
from reconlab.synthetic import PosedImageDataset, SyntheticPolicy, write_input_evidence


def test_completed_inputs_are_not_overwritten_by_renderer(tmp_path):
    write_json(tmp_path / "INPUTS_COMPLETE.json", {"inputs_sha256": "recorded"})
    marker = tmp_path / "source.bin"
    marker.write_bytes(b"preserve me")
    with pytest.raises(FileExistsError, match="immutable"):
        BlenderObjectRenderer(SyntheticPolicy()).render(marker, tmp_path, {}, stage="inputs")
    assert marker.read_bytes() == b"preserve me"


def test_approved_cameras_project_origin_and_keep_box_in_frame():
    policy = SyntheticPolicy()
    data = policy.transforms()
    assert [len(data[f"{s}_filenames"]) for s in ("train", "val", "test")] == [36, 6, 6]
    poses = np.array([f["transform_matrix"] for f in data["frames"]])
    views = opengl_to_view(poses)
    center = views @ np.array([0, 0, 0, 1])
    np.testing.assert_allclose(center[:, :2], 0, atol=1e-6)
    np.testing.assert_allclose(center[:, 2], policy.radius, atol=1e-6)
    # Every corner of the largest possible normalized bounding cube stays visible.
    box = np.array([[x, y, z, 1] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
    coords = np.einsum("nij,kj->nki", views, box)
    uv = coords[..., :2] / coords[..., 2:3] * data["fl_x"] + policy.width / 2
    assert uv.min() > 0 and uv.max() < policy.width
    _, orbit = policy.cameras()
    assert len(orbit["camera_path"]) == 72
    assert orbit["seconds"] == 3


def test_posed_dataset_detects_pixel_and_pose_changes(tmp_path):
    # Small generated RGB fixtures test integrity, not reconstruction quality.
    policy = SyntheticPolicy(width=8, height=8)
    data = policy.transforms()
    (tmp_path / "images").mkdir()
    for f in data["frames"]:
        Image.new("RGB", (8, 8), "white").save(tmp_path / f["file_path"])
    write_json(tmp_path / "transforms.json", data)
    write_input_evidence(tmp_path, policy, {})
    dataset = PosedImageDataset(tmp_path)
    assert dataset.load_split("train")[0].shape == (36, 8, 8, 3)
    first = tmp_path / data["train_filenames"][0]
    Image.new("RGB", (8, 8), "red").save(first)
    with pytest.raises(ValidationError, match="Input changed"):
        dataset.load_split("train")
    data["fl_x"] += 1
    write_json(tmp_path / "transforms.json", data)
    with pytest.raises(ValidationError, match="Cameras or splits changed"):
        PosedImageDataset(tmp_path)


def test_posed_dataset_rejects_split_leakage(tmp_path):
    policy = SyntheticPolicy(width=8, height=8)
    data = policy.transforms()
    data["test_filenames"].append(data["train_filenames"][0])
    write_json(tmp_path / "transforms.json", data)
    write_json(tmp_path / "inputs.json", {})
    with pytest.raises(ValidationError, match="overlap"):
        PosedImageDataset(tmp_path)
