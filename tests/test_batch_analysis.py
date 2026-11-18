import json

import numpy as np
import pytest

from reconlab.batch_analysis import BatchGallery, ForegroundMetrics


def test_foreground_exposes_error_hidden_by_white_background():
    target = np.full((100, 100, 3), 255, np.uint8)
    target[40:50, 40:50] = 0
    prediction = np.full_like(target, 255)
    alpha = np.zeros((100, 100), np.uint8)
    alpha[40:50, 40:50] = 255
    result = ForegroundMetrics.measure(target, prediction, alpha)
    assert result["full_psnr_db"] == pytest.approx(20)
    assert result["foreground_psnr_db"] == pytest.approx(0)
    assert result["visible_colour_recall_proxy"] == 0


def test_empty_foreground_is_unavailable_not_perfect():
    image = np.full((8, 8, 3), 255, np.uint8)
    result = ForegroundMetrics.measure(image, image, np.zeros((8, 8), np.uint8))
    assert result["foreground_psnr_db"] is None
    assert result["visible_colour_recall_proxy"] is None


def test_metric_rejects_misaligned_inputs():
    with pytest.raises(ValueError):
        ForegroundMetrics.measure(np.zeros((8, 8, 3)), np.zeros((4, 4, 3)), np.zeros((8, 8)))


def test_gallery_keeps_all_100_including_failed_and_pending(tmp_path):
    items = [
        {
            "id": str(i),
            "uid": str(i),
            "name": f"Object {i}",
            "category": "fixture",
            "license": "by",
            "author": {"displayName": "fixture"},
            "url": "https://example.com",
        }
        for i in range(100)
    ]
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"objects": items}))
    progress = tmp_path / "progress/99"
    progress.mkdir(parents=True)
    (progress / "source.json").write_text(
        json.dumps({"state": "FAILED", "error": "fixture failure"})
    )
    page = BatchGallery().build(tmp_path, catalog)
    report = json.loads((tmp_path / "batch_summary.json").read_text())
    assert report["summary"]["total"] == 100
    assert report["summary"]["states"] == {"pending": 99, "failed": 1}
    assert report["objects"][-1]["error"].startswith("source:")
    assert page.stat().st_size < 1024**2
    assert 'src="data:' not in page.read_text()
