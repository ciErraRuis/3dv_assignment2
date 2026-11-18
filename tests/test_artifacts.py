from dataclasses import replace

import pytest

from reconlab.artifacts import ArtifactStore
from reconlab.config import PipelineConfig
from reconlab.data import IdentityPreprocessor, JsonDatasetLoader, verify_inputs
from reconlab.errors import ValidationError
from reconlab.experiments import merge
from reconlab.factory import default_registry
from reconlab.interfaces import ViewSelector
from reconlab.pipeline import ReconstructionPipeline


def test_prepared_reload_and_split_integrity(dataset, tmp_path):
    original = IdentityPreprocessor().prepare(
        JsonDatasetLoader().load(dataset)[0], tmp_path / "data"
    )
    loaded = ArtifactStore().load_prepared(original.root)
    assert original == loaded
    (original.root / "train_list.txt").write_text("view_5.png\n")
    with pytest.raises(ValidationError, match="Split file changed"):
        verify_inputs(original)


def test_component_replacement_does_not_leak_options():
    base = {"matcher": {"name": "ratio", "options": {"ratio": 0.8}}}
    change = {"matcher": {"name": "lightglue", "options": {"features": "superpoint"}}}
    assert merge(base, change)["matcher"]["options"] == {"features": "superpoint"}
    assert base["matcher"]["name"] == "ratio"


def test_selector_cannot_relabel_test_images(dataset, tmp_path):
    class LeakingSelector(ViewSelector):
        def select(self, views):
            return (*views[:-1], replace(views[-1], split="train"))

    registry = default_registry()
    registry.register(ViewSelector, "leaking", LeakingSelector)
    config = PipelineConfig.parse({"name": "bad_selector", "selector": {"name": "leaking"}})
    result = ReconstructionPipeline(config, registry).run(
        dataset, tmp_path / "run", until="prepare"
    )
    assert result["status"] == "FAILED"
    assert "unchanged source views" in result["objects"][0]["error"]
