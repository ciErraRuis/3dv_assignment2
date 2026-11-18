"""Owner-side provenance checks and strict separation of RGB subsets and GT."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from reconlab.benchmark.core import Geometry, save_json, sha256


class SourceBundle:
    def __init__(self, item, output):
        self.item=item
        original=Path(item['source48'])
        manifest=original/'inputs.json'
        assert sha256(manifest)==item['source_manifest_sha256']
        self.evidence=json.loads(manifest.read_text())
        self.data=json.loads((original/'transforms.json').read_text())
        assert sha256(original/'transforms.json')==self.evidence['transforms_sha256']
        assert self.evidence['policy']==item['policy']
        expected={r['path']:r for r in item['image_evidence']}
        assert len(expected)==len(self.data['frames'])==48
        assert {s:sum(f['split']==s for f in self.data['frames']) for s in ['train','val','test']}=={'train':36,'val':6,'test':6}
        for frame in self.data['frames']:
            p=original/frame['file_path'];record=expected[frame['file_path']]
            assert frame['split']==record['split'] and sha256(p)==record['sha256']
            with Image.open(p) as image:
                assert image.size==(512,512) and image.mode=='RGB'
        self.root=output/'source48'/item['uid']
        self.root.mkdir(parents=True,exist_ok=True)
        for name in ['inputs.json','transforms.json','orbit.json']:
            (self.root/name).write_bytes((original/name).read_bytes())
        if not (self.root/'images').exists():
            (self.root/'images').symlink_to((original/'images').resolve())
        self.hashes={Path(k).name:v['sha256'] for k,v in expected.items()}
        self.train_names=[Path(f['file_path']).name for f in self.data['frames'] if f['split']=='train']
        self.canonical_orbit=self._common_orbit()

    def _common_orbit(self):
        # Freeze the old centered presentation in GT evaluation coordinates.
        # This metadata is used only after fitting the new Gaussian model.
        geometry=Geometry.load(Path(self.item['baseline_geometry']))
        center=np.median(geometry.c2w[:,:3,3],axis=0)
        norm_scale=float(np.median(np.linalg.norm(geometry.c2w[:,:3,3]-center,axis=1)))
        align=json.loads((Path(self.item['baseline_cases'])/'step_003000/pose_metrics.json').read_text())
        rotation=np.array(align['rotation']);translation=np.array(align['translation']);scale=align['scale']
        old=json.loads(Path(self.item['baseline_orbit']).read_text())
        poses=np.array(old['camera_to_world_opencv']);world=poses.copy()
        world[:,:3,:3]=np.einsum('ij,njk->nik',rotation,poses[:,:3,:3])
        world[:,:3,3]=scale*((poses[:,:3,3]*norm_scale+center)@rotation.T)+translation
        assert world.shape==(72,4,4) and np.isfinite(world).all()
        p=self.root/'common_orbit.json'
        save_json(p,{'camera_to_world_opencv':world.tolist(),'intrinsics':geometry.intrinsics[0].tolist(),
                    'coordinates':'GT world for post-training presentation alignment only',
                    'source_centered_sha256':sha256(Path(self.item['baseline_orbit'])),
                    'frames':72,'fps':24,'seconds':3,'size':[512,512],
                    'same_physical_cameras_and_intrinsics_for_all_view_counts':True})
        return p

    def subset(self, setting, destination):
        images=destination/'input/images';images.mkdir(parents=True,exist_ok=True)
        selected=set(setting.names)
        assert selected<=set(self.train_names)
        records=[]
        for name in setting.names:
            original=self.root/'images'/name;target=images/name
            if not target.exists():target.symlink_to(original)
            assert sha256(target)==self.hashes[name]
            records.append({'name':name,'sha256':self.hashes[name],'source_index':int(name[5:8]),'size':[512,512]})
        save_json(destination/'input/input_manifest.json',{'frames':records,'input_views':setting.count,
                  'sampling':setting.layout,'gt_cameras_present':False,'preprocessing':'original full512 RGB'})
        evaluation=destination/'evaluation';evaluation.mkdir(exist_ok=True)
        filtered={**self.data,'frames':[f for f in self.data['frames'] if f['split']!='train' or Path(f['file_path']).name in selected]}
        assert len(filtered['frames'])==setting.count+12
        save_json(evaluation/'transforms.json',filtered)
        for name in ['images','orbit.json']:
            if not (evaluation/name).exists():(evaluation/name).symlink_to(self.root/name)
        return images,evaluation
