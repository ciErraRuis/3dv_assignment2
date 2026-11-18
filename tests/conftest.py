import json

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(123)
    base = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    views = []
    for i in range(6):
        image = np.roll(base, 5 * i, axis=1).copy()
        image[:8, :8] = i * 30
        path = tmp_path / f"image_{i}.png"
        Image.fromarray(image).save(path)
        mask = np.zeros((240, 320), dtype=np.uint8)
        mask[20:220, 20:300] = 255
        mask_path = tmp_path / f"mask_{i}.png"
        Image.fromarray(mask).save(mask_path)
        views.append(
            {
                "id": f"view_{i}",
                "image": path.name,
                "mask": mask_path.name,
                "split": "train" if i < 4 else ("val" if i == 4 else "test"),
                "azimuth_deg": i * 60,
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "objects": [{"id": "object_01", "views": views}]})
    )
    return manifest
