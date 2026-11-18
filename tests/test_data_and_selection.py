import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from reconlab.data import IdentityPreprocessor, JsonDatasetLoader, read_rgb, verify_inputs
from reconlab.errors import ValidationError
from reconlab.io import read_json
from reconlab.selection import FractionSelector
from reconlab.types import ObjectViews


def test_identity_preserves_pixels_and_evidence(dataset, tmp_path):
    obj = JsonDatasetLoader().load(dataset)[0]
    prepared = IdentityPreprocessor().prepare(obj, tmp_path / "staged")
    for source, actual in zip(obj.views, prepared.views):
        np.testing.assert_array_equal(read_rgb(source.image), read_rgb(actual.image))
    assert len(read_json(prepared.provenance)["views"]) == 6
    verify_inputs(prepared)
    Image.new("RGB", (320, 240)).save(prepared.views[0].image)
    with pytest.raises(ValidationError, match="changed"):
        verify_inputs(prepared)


def test_duplicate_pixels_across_split_are_rejected(dataset, tmp_path):
    obj = JsonDatasetLoader().load(dataset)[0]
    duplicate = tmp_path / "duplicate.png"
    duplicate.write_bytes(obj.views[0].image.read_bytes())
    obj = ObjectViews(obj.id, (*obj.views[:-1], replace(obj.views[-1], image=duplicate)))
    with pytest.raises(ValidationError, match="Duplicate decoded"):
        IdentityPreprocessor().prepare(obj, tmp_path / "duplicate_stage")


@pytest.mark.parametrize("strategy", ["uniform_index", "contiguous", "angular_farthest"])
def test_selectors_preserve_heldout(dataset, strategy):
    obj = JsonDatasetLoader().load(dataset)[0]
    selected = FractionSelector(0.5, strategy).select(obj.views)
    assert [v.id for v in selected if v.split != "train"] == ["view_4", "view_5"]
    assert len([v for v in selected if v.split == "train"]) == 2
    assert selected == FractionSelector(0.5, strategy).select(obj.views)


def test_angles_are_required(dataset):
    obj = JsonDatasetLoader().load(dataset)[0]
    with pytest.raises(ValidationError, match="azimuth"):
        FractionSelector(0.5, "angular_farthest").select(
            [replace(v, azimuth_deg=None) for v in obj.views]
        )


def test_bad_masks_are_not_resized_or_thresholded(dataset, tmp_path):
    obj = JsonDatasetLoader().load(dataset)[0]
    Image.new("L", (320, 240), 127).save(obj.views[0].mask)
    with pytest.raises(ValidationError, match="0/255"):
        IdentityPreprocessor().prepare(obj, tmp_path / "bad_mask")


def test_manifest_unknown_fields_and_traversal(dataset):
    raw = json.loads(dataset.read_text())
    raw["objects"][0]["id"] = "../../oops"
    dataset.write_text(json.dumps(raw))
    with pytest.raises(ValidationError, match="Unsafe"):
        JsonDatasetLoader().load(dataset)


def test_alpha_input_requires_policy(tmp_path):
    image = tmp_path / "alpha.png"
    Image.new("RGBA", (10, 10)).save(image)
    with pytest.raises(ValidationError, match="alpha policy"):
        read_rgb(image)
