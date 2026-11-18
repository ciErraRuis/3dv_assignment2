"""Frozen RoMa experiment: cached ten-object ablation, then fixed 100-object expansion."""
from pathlib import Path
from types import SimpleNamespace
import argparse, datetime, gc, hashlib, importlib.util, json, math, os, shutil, subprocess, sys, time, traceback
import numpy as np
from PIL import Image
from filtering import FilteredMatcher, FilterPolicy, BASE
from reconlab.benchmark.core import Geometry, save_json, sha256

CODE = Path(__file__).resolve().parent
CONFIG = json.loads((CODE.parents[1] / 'configs/roma100.json').read_text())
OUT = Path(CONFIG['output'])
OLD = Path(CONFIG['baseline_root'])
PYTHON = CONFIG['python']

def write(path, value):
    save_json(path, value)

def read(path):
    return json.loads(path.read_text())

def source(uid):
    """Expose exactly 6 train / 6 val / 6 test, keeping original pixel files intact."""
    src = Path(CONFIG['source_root']) / 'evaluation_inputs' / uid
    dst = OUT / 'evaluation_inputs' / uid
    if not (src / 'INPUTS_COMPLETE.json').exists():
        from sources import ensure_source
        src = ensure_source(uid, CONFIG, OUT)
    record = read(src / 'inputs.json')
    if sha256(src / 'transforms.json') != record['transforms_sha256']:
        raise ValueError('Source transform provenance mismatch')
    data = read(src / 'transforms.json')
    selected = set(CONFIG['selected_views'])
    data['frames'] = [f for f in data['frames'] if f['split'] != 'train' or Path(f['file_path']).name in selected]
    assert sum(f['split']=='train' for f in data['frames']) == 6
    assert sum(f['split']=='val' for f in data['frames']) == 6
    assert sum(f['split']=='test' for f in data['frames']) == 6
    dst.mkdir(parents=True, exist_ok=True)
    for key in ['images', 'orbit.json']:
        if not (dst/key).exists(): (dst/key).symlink_to(src/key)
    write(dst/'transforms.json',data)
    write(dst/'inputs.json',{**record,'transforms_sha256':sha256(dst/'transforms.json'),
                            'source_manifest':str(src/'inputs.json'),'source_manifest_sha256':sha256(src/'inputs.json')})
    from reconlab.benchmark.runner import prepare_input
    images = prepare_input(dst, OUT/'inputs'/uid, 6)
    return dst,images

def native():
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    free,total=torch.cuda.mem_get_info()
    if free < 16*1024**3:raise RuntimeError('Less than 16 GiB free GPU memory')
    torch.set_num_threads(4)
    pin=CONFIG['native_binary']
    assert sha256(Path(pin['path']))==pin['sha256']
    spec=importlib.util.spec_from_file_location('gsplat_cuda',pin['path'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.modules['gsplat.csrc']=module
    import gsplat
    gsplat.csrc=module
    from gsplat.cuda._backend import _C
    assert _C is module
    for name in ["fully_fused_projection_fwd", "fully_fused_projection_bwd", "rasterize_to_pixels_fwd", "rasterize_to_pixels_bwd"]:
        assert hasattr(_C,name), name

def centered(geometry,params,center,scale,destination):
    """Presentation path only; independent of heldout metrics and original orbit."""
    import torch
    from gsplat import rasterization
    spec=importlib.util.spec_from_file_location('centered_policy',CODE / 'presentation.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    arrays={k:v.detach().cpu().numpy() for k,v in params.items()}
    poses=geometry.c2w.copy();poses[:,:3,3]=(poses[:,:3,3]-center)/scale
    model=SimpleNamespace(xyz=arrays['means'],scales=np.exp(arrays['scales']),
                          opacity=1/(1+np.exp(-arrays['opacities'][:,None])),
                          sh=np.concatenate([arrays['sh0'],arrays['shN']],1),c2w=poses,k=geometry.intrinsics[0])
    orbit=module.CenteredOrbit(model)
    frames=[];directory=destination/'centered';directory.mkdir(exist_ok=True)
    with torch.inference_mode():
        for i,pose in enumerate(orbit.poses):
            img,_,_=rasterization(means=params['means'],quats=params['quats'],scales=params['scales'].exp(),
                opacities=params['opacities'].sigmoid(),colors=torch.cat([params['sh0'],params['shN']],1),
                viewmats=torch.tensor(np.linalg.inv(pose)[None],dtype=torch.float32,device='cuda'),
                Ks=torch.tensor(model.k[None],dtype=torch.float32,device='cuda'),width=512,height=512,
                sh_degree=2,backgrounds=torch.ones((1,3),device='cuda'),packed=False,near_plane=.01,far_plane=100)
            image=Image.fromarray((img[0].clamp(0,1).cpu().numpy()*255).round().astype(np.uint8))
            image.save(directory/f'frame_{i:05d}.png');frames.append(image)
    durations=[(round((i+1)*100/24)-round(i*100/24))*10 for i in range(72)]
    frames[0].save(destination/'centered.gif',save_all=True,append_images=frames[1:],duration=durations,loop=0,disposal=2)
    write(destination/'centered.json',orbit.metadata)

class Worker:
    def __init__(self,shard):
        self.shard=shard;self.matcher=None
    def case(self,uid,variant,comparison=False):
        import torch
        from reconlab.benchmark.gaussian import GaussianBenchmark
        from reconlab.benchmark.sfm import PairwiseReconstruction,IncrementalMapper
        from reconlab.benchmark.visuals import export_geometry
        from reconlab.benchmark.matchers import RomaV2Matcher
        destination=OUT/('comparison' if comparison else 'results')/variant/uid
        if (destination/'status.json').exists() and read(destination/'status.json')['state'] in ['COMPLETE','FAILED']:return
        destination.mkdir(parents=True,exist_ok=True)
        start=time.monotonic();stage='prepare_source'
        write(destination/'status.json',{'state':'RUNNING','stage':stage,'uid':uid})
        try:
            src,images=source(uid)
            stage='matching_and_sfm'
            write(destination/'status.json',{'state':'RUNNING','stage':stage,'uid':uid})
            if comparison:
                cached=Path(CONFIG['cpu_geometry'])/variant/uid
                status=read(cached/'CPU_STATUS.json')
                if status['state']!='COMPLETE':raise RuntimeError('CPU geometry failed: '+str(status))
                expected=read(OLD/'inputs'/uid/'input_manifest.json')['frames']
                assert read(OUT/'inputs'/uid/'input_manifest.json')['frames']==expected
                shutil.copytree(cached,destination,dirs_exist_ok=True,symlinks=True)
                geometry=Geometry.load(destination)
            else:
                torch.manual_seed(42)
                torch.set_float32_matmul_precision('highest')
                if self.matcher is None:self.matcher=RomaV2Matcher(Path(CONFIG['roma_source']))
                policy=FilterPolicy(foreground=variant!='baseline',minimum_overlap=.5 if variant=='fg_conf050' else 0)
                geometry=PairwiseReconstruction(FilteredMatcher(self.matcher,policy),variant,IncrementalMapper(4)).reconstruct(images,destination)
                geometry.save(destination)
                export_geometry(geometry,src,destination)
            stage='gaussian_3000'
            write(destination/'status.json',{'state':'RUNNING','stage':stage,'uid':uid,'registered':len(geometry.names)})
            gaussian=GaussianBenchmark(3000)
            params,center,scale=gaussian.fit(geometry,images,destination/'gaussian')
            stage='evaluation'
            gaussian.evaluate(geometry,src,destination/'gaussian',params,center,scale)
            try:centered(geometry,params,center,scale,destination/'gaussian')
            except Exception: (destination/'presentation_error.log').write_text(traceback.format_exc())
            del params
            status={'state':'COMPLETE','registered':len(geometry.names),'uid':uid}
        except (AttributeError, ImportError, PermissionError) as error:
            (destination/'error.log').write_text(traceback.format_exc())
            write(destination/'status.json',{'state':'INFRASTRUCTURE_FAILED','error':repr(error),'stage':stage})
            raise
        except Exception as error:
            (destination/'error.log').write_text(traceback.format_exc())
            status={'state':'FAILED','stage':stage,'uid':uid,'error':repr(error)}
        write(destination/'status.json',{**status,'seconds':time.monotonic()-start,'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        gc.collect();torch.cuda.empty_cache()
    def run(self,phase):
        native()
        uids=CONFIG['shards'][self.shard]['uids']
        if phase=='compare':
            for uid in uids:
                if uid in CONFIG['selection_uids']:
                    for variant in ['fg','fg_conf050']:self.case(uid,variant,True)
        else:
            variant=read(OUT/'selection.json')['winner']
            for uid in uids:
                if uid in CONFIG['selection_uids']:
                    dst=OUT/'results'/variant/uid;dst.parent.mkdir(parents=True,exist_ok=True)
                    if not dst.exists():dst.symlink_to(OUT/'comparison'/variant/uid)
                else:self.case(uid,variant)
        write(OUT/'workers'/f'{phase}_{self.shard}.json',{'state':'FINISHED','uids':uids,'phase':phase})

def select():
    scores=[]
    for variant in ['baseline','fg','fg_conf050']:
        values=[];rows=[]
        for uid in CONFIG['selection_uids']:
            directory=OUT/'comparison'/variant/uid
            status=read(directory/'status.json') if (directory/'status.json').exists() else {'state':'MISSING'}
            vals=[]
            if status['state']=='COMPLETE':
                vals=[v['foreground_psnr'] for v in read(directory/'gaussian/render_metrics.json')['views'] if v['split']=='val']
                if len(vals)!=6 or any(v is None or not math.isfinite(v) for v in vals):vals=[]
            mean=float(np.mean(vals)) if vals else None
            if mean is not None:values.append(mean)
            rows.append({'uid':uid,'state':status['state'],'val_foreground_psnr':mean})
        scores.append({'variant':variant,'successful':len(values),'mean_val_foreground_psnr':float(np.mean(values)) if values else None,'cases':rows})
    # Stable max preserves baseline -> fg -> fg_conf050 order for exact ties.
    winner=max(scores,key=lambda x:(x['successful'],x['mean_val_foreground_psnr'] if x['mean_val_foreground_psnr'] is not None else -math.inf))
    if winner['successful']==0:raise RuntimeError('No successful validation cases; expansion prohibited')
    write(OUT/'selection.json',{'winner':winner['variant'],'scores':scores,'test_used':False,'selection_objects_report_separately':True})
    return winner['variant']

def run_phase(phase):
    children=[]
    for shard in range(4):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(shard))
        log=(OUT/'logs'/f'{phase}_{shard}.log').open('w')
        child=subprocess.Popen([PYTHON,'-u',str(CODE/'runtime.py'),'worker',phase,str(shard)],env=env,stdout=log,stderr=subprocess.STDOUT)
        children.append((child,log))
    while any(p.poll() is None for p,_ in children):
        from report import build
        build(OUT)
        write(OUT/'heartbeat.json',{'phase':phase,'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'workers':[p.poll() for p,_ in children]})
        time.sleep(30)
    codes=[p.wait() for p,_ in children]
    for _,log in children:log.close()
    if any(codes):raise RuntimeError(f'{phase} worker failure: {codes}')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['logs','workers']: (OUT/name).mkdir(exist_ok=True)
    if sys.argv[1]=='worker':Worker(int(sys.argv[3])).run(sys.argv[2]);return
    # Worker starts only once the predecessor has released its four GPUs.
    previous=Path(CONFIG['source_root'])/'execution_complete.json'
    # User stopped the predecessor; missing source views are produced per case.
    # No MASt3R reconstruction is resumed.
    if not (Path(CONFIG['cpu_geometry']).parent/'CPU_BATCH_COMPLETE.json').exists():raise RuntimeError('CPU comparison incomplete')
    inventory=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.free','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout
    write(OUT/'resource_check.json',{'gpus':inventory,'time':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    if len(inventory.strip().splitlines())!=4:raise RuntimeError('Expected exactly four allocated GPUs')
    prior=Path(CONFIG['reuse_comparison_root'])
    old_selection=read(prior/'selection.json')
    for variant in ['baseline','fg','fg_conf050']:
        dst=OUT/'comparison'/variant;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():dst.symlink_to(prior/'comparison'/variant)
    write(OUT/'selection.json',{**old_selection,'winner':CONFIG['forced_variant'],
        'selection_basis':'Explicit user choice after reviewing comparison; overrides coverage-first automaticselection',
        'previous_automatic_winner':old_selection['winner'],
        'threshold':0.5,'fallback_to_rejected_points':False})
    run_phase('expand')
    from report import build
    build(OUT)
    write(OUT/'execution_complete.json',{'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':'EXECUTION_COMPLETE','note':'Individual failures preserved; see status.json per object'})

if __name__=='__main__':main()
