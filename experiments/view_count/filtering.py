"""Post-match filtering; source RGB and official model preprocessing stay unchanged."""
from dataclasses import dataclass
from pathlib import Path
import json,sys,hashlib
import numpy as np
from PIL import Image
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE/'src'))
from reconlab.benchmark.core import PairMatches,save_json,sha256

@dataclass(frozen=True)
class FilterPolicy:
    foreground: bool = True
    minimum_overlap: float = 0.0
    white_floor: int = 250
    foreground_margin: int = 1

class BorderWhiteForeground:
    def __init__(self,policy):self.policy=policy
    def __call__(self,path):
        import cv2
        rgb=np.asarray(Image.open(path).convert('RGB'))
        near_white=(rgb>=self.policy.white_floor).all(-1).astype(np.uint8)
        _,labels=cv2.connectedComponents(near_white,connectivity=4)
        boundary=np.unique(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]])
        boundary=boundary[boundary!=0]
        foreground=~np.isin(labels,boundary)
        if self.policy.foreground_margin:
            k=2*self.policy.foreground_margin+1
            foreground=cv2.dilate(foreground.astype(np.uint8),np.ones((k,k),np.uint8)).astype(bool)
        return foreground

def inside(points,mask):
    xy=np.rint(points).astype(int);h,w=mask.shape
    valid=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    good=np.zeros(len(xy),bool);good[valid]=mask[xy[valid,1],xy[valid,0]]
    return good

class FilteredMatcher:
    def __init__(self,raw,policy):self.raw=raw;self.policy=policy;self.masks={};self.reused=0
    def prepare(self,paths):
        self.raw.evidence_root=self.evidence_root
        self.raw.prepare(paths)
        self.display_paths=getattr(self.raw,'display_paths',{})
        if self.policy.foreground:
            masks=self.evidence_root.parent/'foreground_masks';masks.mkdir(parents=True,exist_ok=True)
            for p in paths:
                self.masks[p]=BorderWhiteForeground(self.policy)(p)
                Image.fromarray(self.masks[p].astype(np.uint8)*255).save(masks/p.name)
    def match(self,a,b):
        m=self.raw.match(a,b);fg=np.ones(len(m.scores),bool)
        if self.policy.foreground:fg=inside(m.first,self.masks[a]) & inside(m.second,self.masks[b])
        confidence=m.scores>=self.policy.minimum_overlap
        keep=fg & confidence
        audit=self.evidence_root.parent/'filter_audit';audit.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(audit/f'{a.stem}__{b.stem}.npz',first=m.first,second=m.second,scores=m.scores,foreground_pass=fg,confidence_pass=confidence,kept=keep)
        save_json(audit/f'{a.stem}__{b.stem}.json',{'raw':len(keep),'foreground_pass':int(fg.sum()),'score_pass':int(confidence.sum()),'kept':int(keep.sum()),'minimum_overlap':self.policy.minimum_overlap,'score_semantics':'predicted overlap, not calibrated correspondence correctness','fallback_to_rejected_points':False})
        self.reused=getattr(self.raw,'reused',0)
        self.display_paths=getattr(self.raw,'display_paths',{})
        return PairMatches(m.first[keep],m.second[keep],m.scores[keep])

class FrozenPairs:
    def __init__(self,root,uid):self.root=Path(root);self.uid=uid;self.reused=0
    def prepare(self,paths):
        manifest=json.loads((self.root/'inputs'/self.uid/'input_manifest.json').read_text())
        expected={x['name']:x['sha256'] for x in manifest['frames']}
        for p in paths:
            if sha256(p)!=expected[p.name]:raise ValueError('Cachedinputchanged:'+str(p))
        self.display_paths={};self.source=self.root/'results/romav2'/self.uid
        for stage in ['coarse_800','fine_1280']:
            for p in paths:
                src=self.source/'model_inputs'/stage/p.name
                if src.exists():
                    dst=self.evidence_root/stage/p.name;dst.parent.mkdir(parents=True,exist_ok=True)
                    if not dst.exists():dst.symlink_to(src)
                    info=src.with_suffix('.json')
                    if info.exists() and not dst.with_suffix('.json').exists():dst.with_suffix('.json').symlink_to(info)
                    self.display_paths[p]=dst
    def match(self,a,b):
        p=self.source/'matches'/f'{a.stem}__{b.stem}.npz'
        with np.load(p) as d:m=PairMatches(d['first'],d['second'],d['scores'])
        self.reused+=1
        save_json(self.evidence_root.parent/'raw_pair_provenance'/f'{a.stem}__{b.stem}.json',{'source':str(p),'sha256':sha256(p),'reuse':'raw model correspondences only; RANSAC and SfM rerun'})
        return m
