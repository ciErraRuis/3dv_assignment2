"""Geometry and leakage checks for the image-only benchmark."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from reconlab.benchmark.core import PairMatches, TrackGraph, sha256, similarity_alignment
from reconlab.benchmark.runner import prepare_input
from reconlab.benchmark.view_ablation import ViewSubset


def test_roma_restores_required_precision_after_dependency_initialization(monkeypatch, tmp_path):
    from reconlab.benchmark.matchers import RomaV2Matcher

    state = {"precision": "highest"}

    class NativeModel:
        Cfg = SimpleNamespace

        def __init__(self, config):
            state["precision"] = "high"

        def register_forward_pre_hook(self, hook, with_kwargs):
            assert state["precision"] == "highest"

    monkeypatch.setitem(sys.modules, "romav2", SimpleNamespace(RoMaV2=NativeModel))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(set_float32_matmul_precision=lambda value: state.update(precision=value)),
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    RomaV2Matcher(tmp_path)
    assert state["precision"] == "highest"


def test_view_subsets_preserve_counts_and_sampling_scope():
    names = [f"view_{i:03d}.png" for i in range(36)]
    for count in [3, 5, 6, 10, 20, 36]:
        uniform = ViewSubset.select(names, count, "uniform_orbit")
        adjacent = ViewSubset.select(names, count, "contiguous_arc")
        assert len(set(uniform)) == len(set(adjacent)) == count
        assert adjacent == names[:count]
        assert uniform == [names[int(i * 36 / count)] for i in range(count)]
    assert ViewSubset.select(names, 36, "uniform_orbit") == names


def test_similarity_alignment_recovers_gauge():
    source = np.random.default_rng(2).normal(size=(20, 3))
    rotation = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    target = 2.5 * source @ rotation.T + np.array([3.0, 4, 7])
    s, r, t = similarity_alignment(source, target)
    np.testing.assert_allclose(s * source @ r.T + t, target, atol=1e-10)
    assert np.linalg.det(r) > 0


def test_tracks_share_identity_across_pairs_and_reject_invalid_pixels():
    graph = TrackGraph(["a", "b", "c"])
    graph.add(
        "a", "b", PairMatches([[10.1, 20.1], [-1, 1]], [[30, 40], [3, 4]], [1, 1]), np.ones(2, bool)
    )
    graph.add("b", "c", PairMatches([[30.2, 40.1]], [[50, 60]], [1]), np.ones(1, bool))
    assert graph.pairs[("a", "b")][0, 1] == graph.pairs[("b", "c")][0, 0]
    assert len(graph.pairs[("a", "b")]) == 1


def test_input_manifest_excludes_cameras_and_heldout(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "images").mkdir()
    frames, records = [], []
    for i in range(48):
        name = f"images/view_{i:03d}.png"
        p = source / name
        Image.new("RGB", (512, 512), (i, 20, 30)).save(p)
        split = "train" if i < 36 else "val" if i < 42 else "test"
        frames.append({"file_path": name, "split": split, "transform_matrix": np.eye(4).tolist()})
        records.append({"path": name, "sha256": sha256(p)})
    (source / "transforms.json").write_text(json.dumps({"frames": frames}))
    (source / "inputs.json").write_text(
        json.dumps({"images": records, "transforms_sha256": sha256(source / "transforms.json")})
    )
    dest = tmp_path / "algorithm_input"
    image_dir = prepare_input(source, dest)
    assert len(list(image_dir.iterdir())) == 36
    assert not (dest / "transforms.json").exists()
    assert not (image_dir / "view_036.png").exists()
    from reconlab.benchmark.sparse import SparseDataset

    sparse = tmp_path / "sparse"
    selected = SparseDataset.create(source, sparse)
    sparse_images = prepare_input(sparse, tmp_path / "sparse_algorithm", 6)
    assert selected["pool_indices"] == [0, 6, 12, 18, 24, 30]
    assert sorted(p.name for p in sparse_images.iterdir()) == selected["training_names"]
    assert len(json.loads((sparse / "inputs.json").read_text())["images"]) == 18
    assert not list(sparse_images.glob("*.json"))
    (source / "images/view_000.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed RGB"):
        prepare_input(source, dest)


def test_short_gaussian_schedule_scales_strategy(monkeypatch):
    from reconlab.benchmark.rendering import GaussianRenderer

    class Strategy:
        def __init__(self, **kwargs):
            self.refine_start_iter = 500
            self.refine_stop_iter = 15000
            self.refine_every = 100
            self.reset_every = 3000
            self.noise_injection_stop_iter = -1

    monkeypatch.setitem(
        sys.modules,
        "gsplat.strategy",
        SimpleNamespace(DefaultStrategy=Strategy, MCMCStrategy=Strategy),
    )
    short = GaussianRenderer("2dgs", 0.1).strategy()
    full = GaussianRenderer("2dgs").strategy()
    assert (
        short.refine_start_iter,
        short.refine_stop_iter,
        short.refine_every,
        short.reset_every,
    ) == (50, 1500, 10, 300)
    assert full.refine_stop_iter == 15000
    assert short.noise_injection_stop_iter == -1


def test_estimated_scene_training_does_not_use_gt_poses(tmp_path):
    from reconlab.benchmark.core import Geometry
    from reconlab.benchmark.estimated_scene import EstimatedScene

    names = [f"view_{i:03d}.png" for i in range(6)]
    poses = np.repeat(np.eye(4)[None], 6, axis=0)
    poses[:, :3, 3] = np.random.default_rng(42).normal(size=(6, 3))
    k = np.repeat(np.diag([500.0, 500.0, 1.0])[None], 6, axis=0)
    geometry = Geometry(
        names,
        poses,
        k,
        np.random.default_rng(7).normal(size=(30, 3)),
        np.zeros((30, 3), np.uint8),
        {},
    )
    data = {
        "frames": [
            {"file_path": n, "split": "train", "transform_matrix": (np.eye(4) * 99).tolist()}
            for n in names
        ],
        "fl_x": 999,
        "fl_y": 999,
        "cx": 256,
        "cy": 256,
    }
    (tmp_path / "transforms.json").write_text(json.dumps(data))
    (tmp_path / "inputs.json").write_text(
        json.dumps({"images": [], "transforms_sha256": sha256(tmp_path / "transforms.json")})
    )
    scene = EstimatedScene(tmp_path, geometry)
    assert scene._alignment is None
    np.testing.assert_array_equal(scene.train_intrinsics, k)
    expected = poses.copy()
    expected[:, :3, 3] = (poses[:, :3, 3] - scene.center) / scene.scale
    actual = np.stack([scene.view(f["transform_matrix"]) for f in scene.frames["train"]])
    np.testing.assert_allclose(actual, np.linalg.inv(expected), atol=1e-6)


def test_pair_cache_rejects_changed_inputs(tmp_path):
    from reconlab.benchmark.sparse import CachedPairMatcher

    path = tmp_path / "source.png"
    path.write_bytes(b"changed")
    root = tmp_path / "inputs/object"
    root.mkdir(parents=True)
    (root / "input_manifest.json").write_text(
        json.dumps({"frames": [{"name": "source.png", "sha256": "old"}]})
    )
    matcher = CachedPairMatcher(
        "sift", {"object_uid": "object", "pair_cache_root": str(tmp_path)}, tmp_path, tmp_path
    )
    matcher.evidence_root = tmp_path / "evidence"
    with pytest.raises(ValueError, match="Pair cache input differs"):
        matcher.prepare([path])


def test_sparse_selection_uses_validation_coverage_and_blocks_missing(tmp_path):
    from reconlab.benchmark.core import save_json
    from reconlab.benchmark.sparse_scheduler import ValidationSelector

    objects = [{"uid": "a"}, {"uid": "b"}]
    for method, scores in [("reliable", [15.0, 16.0]), ("partial", [30.0, None])]:
        for obj, score in zip(objects, scores):
            p = tmp_path / "results" / method / obj["uid"]
            save_json(p / "status.json", {"state": "COMPLETE" if score else "FAILED"})
            save_json(p / "val_metrics.json", {"foreground_psnr": score})
            (p / "test_metrics.json").write_text("TEST MUST NOT BE READ")
    selector = ValidationSelector()
    result = selector.rank(tmp_path, ["partial", "reliable"], objects)
    assert result["winner"] == "reliable"
    assert result["ranking"][0]["mean_val_foreground_psnr"] == 15.5
    assert selector.rank(tmp_path, ["partial", "reliable", "pending"], objects)["winner"] is None
