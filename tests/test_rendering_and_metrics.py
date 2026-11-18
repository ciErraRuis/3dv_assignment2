import numpy as np
import pytest
from PIL import Image

from reconlab.errors import ValidationError
from reconlab.evaluation import PSNRMetric
from reconlab.io import read_json
from reconlab.rendering import GifExporter, OrbitCameraPath, validate_camera_path
from reconlab.types import RenderArtifact


def test_orbit_geometry():
    result = OrbitCameraPath().build((0, 0, 0), 2, 15, 12, 24, 320, 240)
    validate_camera_path(result)
    for frame in result["camera_path"]:
        matrix = np.array(frame["camera_to_world"]).reshape(4, 4)
        np.testing.assert_allclose(np.linalg.norm(matrix[:3, 3]), 2)
        np.testing.assert_allclose(matrix[:3, 2], matrix[:3, 3] / 2)
    assert result["seconds"] == 0.5


def test_invalid_camera_rejected():
    result = OrbitCameraPath().build((0, 0, 0), 2, 15, 12, 24, 320, 240)
    result["camera_path"][0]["camera_to_world"][0] = 99
    with pytest.raises(ValidationError, match="rigid"):
        validate_camera_path(result)


def test_gif_duration_and_provenance(tmp_path):
    paths = []
    for i in range(12):
        path = tmp_path / f"frame_{i:03d}.png"
        Image.new("RGB", (320, 240), (i * 20, 0, 0)).save(path)
        paths.append(path)
    output = GifExporter().export(
        RenderArtifact(tuple(paths), 24, {"kind": "synthetic_test"}), tmp_path / "test.gif"
    )
    with Image.open(output) as gif:
        assert gif.n_frames == 12
        durations = []
        for i in range(gif.n_frames):
            gif.seek(i)
            durations.append(gif.info["duration"])
        assert sum(durations) == 500
    assert len(read_json(output.with_suffix(".gif.json"))["source_frames"]) == 12


def test_psnr_known_error_and_mask():
    reference = np.zeros((10, 10, 3), dtype=np.uint8)
    prediction = np.full_like(reference, 255)
    assert PSNRMetric().compute(prediction, reference)["psnr_db"] == 0
    assert PSNRMetric().compute(reference, reference)["perfect_match"]
    prediction[:5] = 0
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5] = True
    assert PSNRMetric().compute(prediction, reference, mask)["perfect_match"]
    with pytest.raises(ValidationError, match="mask"):
        PSNRMetric().compute(prediction, reference, np.zeros_like(mask))


def test_gif_refuses_resize(tmp_path):
    paths = [tmp_path / "a.png", tmp_path / "b.png"]
    for path, size in zip(paths, [(20, 30), (30, 20)]):
        Image.new("RGB", size).save(path)
    with pytest.raises(ValidationError, match="identical dimensions"):
        GifExporter().export(RenderArtifact(tuple(paths), 24), tmp_path / "bad.gif")
