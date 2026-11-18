from dataclasses import replace

import pytest

from reconlab.backends.nerfstudio import NerfactoReconstructor, SplatfactoReconstructor
from reconlab.data import IdentityPreprocessor, JsonDatasetLoader
from reconlab.errors import ValidationError
from reconlab.types import SfMArtifact


def test_nerfstudio_commands_are_explicit(dataset, tmp_path):
    obj = IdentityPreprocessor().prepare(
        JsonDatasetLoader().load(dataset)[0], tmp_path / "prepared"
    )
    sfm = SfMArtifact(tmp_path / "sfm", tuple(v.image.name for v in obj.views), {}, False)
    for cls in (NerfactoReconstructor, SplatfactoReconstructor):
        command = cls().build_command(obj, sfm, tmp_path / "training")
        assert command[command.index("--downscale-factor") + 1] == "1"
        assert command[command.index("--pipeline.model.camera-optimizer.mode") + 1] == "off"
        assert str(sfm.root) in command
        assert "--save-only-latest-checkpoint" in command
    with pytest.raises(ValidationError, match="undistortion"):
        SplatfactoReconstructor().build_command(
            obj, replace(sfm, requires_undistortion=True), tmp_path / "training"
        )
    with pytest.raises(ValidationError, match="registration"):
        NerfactoReconstructor().build_command(
            obj, replace(sfm, registered_names=sfm.registered_names[:-1]), tmp_path / "training"
        )
