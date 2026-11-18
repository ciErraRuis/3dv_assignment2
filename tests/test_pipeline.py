from pathlib import Path

import h5py
import pytest

from reconlab.config import PipelineConfig
from reconlab.errors import ValidationError
from reconlab.experiments import ExperimentSuite
from reconlab.factory import default_registry
from reconlab.interfaces import FeatureExtractor
from reconlab.io import read_json
from reconlab.pipeline import ReconstructionPipeline


def test_matching_pipeline_and_report(dataset, tmp_path):
    output = tmp_path / "run"
    config = PipelineConfig.parse(
        {"name": "cpu", "extractor": {"name": "sift", "options": {"max_keypoints": 300}}}
    )
    result = ReconstructionPipeline(config).run(dataset, output)
    assert result["status"] == "STAGE_COMPLETE"
    assert result["objects"][0]["status"] == "MATCHED"
    with h5py.File(output / "object_01/matching/matches.h5") as store:
        assert "view_0.png/view_1.png/matches0" in store
    assert "Actual staged model inputs" in (output / "report.html").read_text()
    assert not (output / "object_01/training").exists()
    assert read_json(output / "environment.json")["source_hashes"]
    with pytest.raises(FileExistsError):
        ReconstructionPipeline(config).run(dataset, output)


def test_backend_failure_recorded(dataset, tmp_path):
    class BrokenExtractor(FeatureExtractor):
        def extract(self, view):
            raise RuntimeError("intentional test failure")

    registry = default_registry()
    registry.register(FeatureExtractor, "broken", BrokenExtractor)
    config = PipelineConfig.parse({"name": "failure", "extractor": {"name": "broken"}})
    result = ReconstructionPipeline(config, registry).run(dataset, tmp_path / "failed")
    assert result["status"] == "FAILED"
    assert "intentional test failure" in result["objects"][0]["error"]
    assert (tmp_path / "failed/report.html").exists()


def test_all_four_suites_expand(tmp_path):
    root = Path(__file__).resolve().parent / "fixtures/suites"
    counts = []
    for path in sorted(root.glob("*.json")):
        suite = ExperimentSuite.load(path)
        counts.append(len(suite.expand()))
        target = tmp_path / path.stem
        suite.write_plan(target)
        assert read_json(target / "plan.json")["status"] == "DRAFT_NOT_EXECUTED"
        for config in suite.expand():
            PipelineConfig.load(target / f"{config.name}.json")
    assert counts == [2, 5, 3, 2]


def test_unknown_config_rejected():
    with pytest.raises(ValidationError, match="unknown"):
        PipelineConfig.parse({"name": "oops", "reszie": 512})


def test_registry_type_contract():
    with pytest.raises(ValidationError, match="does not implement"):
        default_registry().register(FeatureExtractor, "invalid", dict)
