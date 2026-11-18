"""Reuse raw RGB correspondences only; every subset reconstructs its own geometry."""
from pathlib import Path
import hashlib
import json
import numpy as np
from reconlab.benchmark.core import PairMatches, save_json, sha256


class CachedPairs:
    def __init__(self, raw, bundle, root, checkpoint_hash):
        self.raw, self.bundle, self.root = raw, bundle, root
        self.checkpoint_hash = checkpoint_hash
        self.reused = 0
        (root/'pairs').mkdir(parents=True, exist_ok=True)
        self.model_inputs = root/'model_inputs'

    def prepare(self, paths):
        self.reused = 0
        self.raw.evidence_root = self.model_inputs
        self.raw.prepare(paths)
        self.display_paths = {}
        for path in paths:
            assert path.name in self.bundle.train_names
            assert sha256(path) == self.bundle.hashes[path.name]
            for stage in ['coarse_800', 'fine_1280']:
                global_dir = self.model_inputs/stage
                global_dir.mkdir(parents=True, exist_ok=True)
                case_dir = self.evidence_root/stage
                case_dir.mkdir(parents=True, exist_ok=True)
                for name in [path.name, path.stem+'.json']:
                    old = Path(self.bundle.item['old_model_inputs'])/stage/name
                    cached = global_dir/name
                    if old.exists() and not cached.exists():
                        if name.endswith('.json'):
                            assert json.loads(old.read_text())['source_sha256'] == self.bundle.hashes[path.name]
                        cached.symlink_to(old)
                    target = case_dir/name
                    if not target.is_symlink() and not target.exists():
                        target.symlink_to(cached)
            self.display_paths[path] = self.model_inputs/'fine_1280'/path.name
        # Do not let the capture hook overwrite a symlink to frozen old evidence.
        self.raw.captured = {p for p in paths if all((self.model_inputs/s/p.name).exists()
                             for s in ['coarse_800', 'fine_1280'])}

    def match(self, a, b):
        key = a.stem+'__'+b.stem
        path = self.root/'pairs'/(key+'.npz')
        meta = path.with_suffix('.json')
        identity = {'images': [self.bundle.hashes[a.name], self.bundle.hashes[b.name]],
                    'checkpoint': self.checkpoint_hash, 'samples': 5000}
        if path.exists():
            info = json.loads(meta.read_text())
            assert info['identity'] == identity and sha256(path) == info['sha256']
            self.reused += 1
        else:
            old = Path(self.bundle.item['old_raw'])/(key+'.npz')
            if old.exists():
                with np.load(old) as data:
                    result = PairMatches(data['first'], data['second'], data['scores'])
                origin = {'kind': 'existing_raw5000_before_filter', 'path': str(old), 'sha256': sha256(old)}
                self.reused += 1
            else:
                import torch
                seed = int(hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:8],16)
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    result = self.raw.match(a,b)
                origin = {'kind': 'new_inference', 'seed': seed}
            temporary = path.with_suffix('.pending')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, first=result.first, second=result.second, scores=result.scores)
            temporary.replace(path)
            save_json(meta, {'identity': identity, 'origin': origin, 'sha256': sha256(path)})
        save_json(self.evidence_root.parent/'raw_pair_provenance'/(key+'.json'),
                  {'cache':str(path),'metadata':str(meta),'reuse':'RGB predictions only; RANSAC, tracks and SfM rebuilt'})
        with np.load(path) as data:
            return PairMatches(data['first'], data['second'], data['scores'])
