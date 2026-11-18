"""RGB images -> RoMa matches -> independent SfM -> optimized3DGS -> centered GIF.

The bundled Doll images contain no camera parameters. --check-inputs is CPU-only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from PIL import Image
from reconlab.benchmark.core import save_json, sha256

ROMA_SHA256 = '1557dec0d21b62366465f7ff4d5fdf228cc695d0582e196ad2b80e05230828b7'
DINO_REPOSITORY = 'facebookresearch_dinov3_adc254450203739c8149213a7a69d8d905b4fcfa'


@dataclass(frozen=True)
class InferenceInputs:
    directory: Path
    images: tuple[Path, ...]
    hashes: dict[str, str]

    @classmethod
    def inspect(cls, directory: Path):
        directory = directory.resolve(strict=True)
        images = tuple(sorted(directory.glob('*.png')))
        if len(images) < 3:
            raise ValueError('At least3 RGB PNG images are required')
        hashes = {}
        for path in images:
            with Image.open(path) as image:
                if image.size != (512, 512) or image.mode != 'RGB':
                    raise ValueError(f'Expected full512x512 RGB input; no automatic resizing: {path}')
            hashes[path.name] = sha256(path)
        manifest = directory.parent / 'manifest.json'
        if manifest.exists():
            records = json.loads(manifest.read_text())['frames']
            expected = {Path(row['file']).name: row['sha256'] for row in records}
            if hashes != expected:
                raise ValueError('Input set or image hashes differ from asset manifest')
        return cls(directory, images, hashes)

    def stage(self, output: Path):
        target = output / 'input/images'
        target.mkdir(parents=True)
        for source in self.images:
            shutil.copyfile(source, target / source.name)
            if sha256(target / source.name) != self.hashes[source.name]:
                raise ValueError('Input changed during staging: ' + str(source))
        save_json(output / 'input/manifest.json', self.record())
        return target

    def record(self):
        return {'images': [{'name': p.name, 'sha256': self.hashes[p.name]} for p in self.images],
                'count': len(self.images), 'size': [512, 512],
                'processing': 'Full original RGB; no crop, stretch or resize',
                'camera_metadata_consumed': False}


class InferenceRuntime:
    def __init__(self, roma_source: Path, torch_home: Path, extension: Path | None):
        self.roma_source, self.torch_home, self.extension = roma_source, torch_home, extension

    def prepare(self):
        if not (self.roma_source / 'src/romav2/romav2.py').is_file():
            raise FileNotFoundError('Provide the pinned RoMaV2 source checkout with --roma-source')
        checkpoint = self.torch_home / 'hub/checkpoints/romav2.0.1.pt'
        if not checkpoint.is_file() or sha256(checkpoint) != ROMA_SHA256:
            raise ValueError('Missing or changed RoMa v2.0.1 checkpoint in --torch-home')
        if not (self.torch_home / 'hub' / DINO_REPOSITORY / 'hubconf.py').is_file():
            raise FileNotFoundError('Pinned DINOv3 source must already exist in the Torch Hub cache')
        os.environ['TORCH_HOME'] = str(self.torch_home)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Inference requires a CUDA GPU; use --check-inputs for CPU validation')
        torch.cuda.set_device(0)
        free, total = torch.cuda.mem_get_info()
        if free < 16 * 1024**3:
            raise RuntimeError('The precise-mode pipeline requires at least16GiB free GPU memory')
        torch.set_num_threads(4)
        torch.hub.set_dir(str(self.torch_home / 'hub'))
        torch.manual_seed(42)
        if self.extension:
            extension = self.extension.resolve(strict=True)
            spec = importlib.util.spec_from_file_location('gsplat_cuda', extension)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules['gsplat.csrc'] = module
            import gsplat
            gsplat.csrc = module
            from gsplat.cuda._backend import _C
            if _C is not module:
                raise RuntimeError('gsplat did not load the selected CUDA extension')
        return {'device': torch.cuda.get_device_name(), 'free_bytes': free, 'total_bytes': total,
                'roma_checkpoint': str(checkpoint), 'roma_sha256': ROMA_SHA256,
                'native_extension': str(self.extension) if self.extension else 'installed gsplat build',
                'native_sha256': sha256(self.extension) if self.extension else None}


class GaussianInferencePipeline:
    def __init__(self, runtime: InferenceRuntime, steps: int):
        self.runtime, self.steps = runtime, steps

    def run(self, inputs: InferenceInputs, output: Path):
        if output.exists():
            raise FileExistsError('Choose a new output directory: ' + str(output))
        resources = self.runtime.prepare()
        sys.path.insert(0, str(ROOT / 'experiments/roma100'))
        from filtering import FilterPolicy, FilteredMatcher
        # Reuse the exact rendering implementation that produced the showcase.
        spec = importlib.util.spec_from_file_location('roma_submission_runtime', ROOT / 'experiments/roma100/runtime.py')
        renderer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(renderer)
        from reconlab.benchmark.gaussian import GaussianBenchmark
        from reconlab.benchmark.matchers import RomaV2Matcher
        from reconlab.benchmark.sfm import PairwiseReconstruction

        output.mkdir(parents=True, exist_ok=False)
        save_json(output / 'inference.json', {'resources': resources, 'steps': self.steps,
                  'seed': 42, 'threshold': .5, 'white_floor': 250, 'mask_margin': 1,
                  'input': inputs.record(), 'gt_used': False,
                  'gif': {'frames': 72, 'fps': 24, 'seconds': 3, 'size': [512, 512]}})
        try:
            images = inputs.stage(output)
            save_json(output / 'status.json', {'state': 'RUNNING', 'stage': 'matching_sfm'})
            matcher = FilteredMatcher(RomaV2Matcher(self.runtime.roma_source), FilterPolicy(True, .5, 250, 1))
            geometry = PairwiseReconstruction(matcher, 'romav2_fg_conf050').reconstruct(images, output)
            geometry.save(output)
            save_json(output / 'status.json', {'state': 'RUNNING', 'stage': 'gaussian_training',
                      'registered': len(geometry.names), 'input_views': len(inputs.images)})
            model = GaussianBenchmark(self.steps)
            params, center, scale = model.fit(geometry, images, output / 'gaussian')
            save_json(output / 'status.json', {'state': 'RUNNING', 'stage': 'rendering'})
            renderer.centered(geometry, params, center, scale, output / 'gaussian')
            save_json(output / 'status.json', {'state': 'COMPLETE', 'steps': self.steps,
                      'registered': len(geometry.names), 'input_views': len(inputs.images),
                      'gif': str(output / 'gaussian/centered.gif'), 'gt_used': False})
        except Exception as error:
            (output / 'error.log').write_text(traceback.format_exc())
            save_json(output / 'status.json', {'state': 'FAILED', 'error': repr(error)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=ROOT / 'assets/doll/images')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--roma-source', type=Path)
    parser.add_argument('--torch-home', type=Path)
    parser.add_argument('--gsplat-extension', type=Path, help='Optional compatible prebuilt gsplat CUDA library')
    parser.add_argument('--steps', type=int, choices=[3000, 15000], default=3000)
    parser.add_argument('--check-inputs', action='store_true', help='Validate RGBs and hashes on CPU, without creating output')
    options = parser.parse_args()
    inputs = InferenceInputs.inspect(options.images)
    if options.check_inputs:
        print(json.dumps(inputs.record(), indent=2))
        return
    for name in ['output', 'roma_source', 'torch_home']:
        if getattr(options, name) is None:
            parser.error('--' + name.replace('_', '-') + ' is required for reconstruction')
    runtime = InferenceRuntime(options.roma_source.resolve(), options.torch_home.resolve(), options.gsplat_extension)
    GaussianInferencePipeline(runtime, options.steps).run(inputs, options.output.resolve())


if __name__ == '__main__':
    main()
