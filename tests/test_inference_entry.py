"""CPU contracts for the image-only demo; not a GPU reconstruction evaluation."""
import importlib.util
from pathlib import Path
import shutil
import sys

from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('doll_inference_entry', ROOT / 'scripts/infer.py')
entry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = entry
spec.loader.exec_module(entry)


def test_bundled_input_staging_preserves_pixels_and_hashes(tmp_path):
    inputs = entry.InferenceInputs.inspect(ROOT / 'assets/doll/images')
    assert len(inputs.images) == 6
    staged = inputs.stage(tmp_path)
    assert {p.name: entry.sha256(p) for p in staged.glob('*.png')} == inputs.hashes
    assert not list(staged.glob('*.json'))


def test_asset_tampering_is_rejected(tmp_path):
    shutil.copytree(ROOT / 'assets/doll', tmp_path / 'doll')
    image = tmp_path / 'doll/images/view_001.png'
    with Image.open(image) as original:
        changed = original.copy()
    changed.putpixel((0, 0), (0, 0, 0))
    changed.save(image)
    with pytest.raises(ValueError, match='hashes differ'):
        entry.InferenceInputs.inspect(image.parent)


def test_existing_output_rejected_before_gpu_access(tmp_path):
    class NoGpuAccess:
        def prepare(self):
            raise AssertionError('Existing output must be rejected before GPU access')

    inputs = entry.InferenceInputs.inspect(ROOT / 'assets/doll/images')
    with pytest.raises(FileExistsError):
        entry.GaussianInferencePipeline(NoGpuAccess(), 3000).run(inputs, tmp_path)


def test_unapproved_dimensions_are_not_implicitly_resized(tmp_path):
    for index in range(3):
        Image.new('RGB', (64, 64)).save(tmp_path / f'{index}.png')
    with pytest.raises(ValueError, match='no automatic resizing'):
        entry.InferenceInputs.inspect(tmp_path)
