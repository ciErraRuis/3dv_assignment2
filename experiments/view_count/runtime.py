"""Independent view-count reconstructions and fixed-budget Gaussian optimization."""
from pathlib import Path
import datetime, gc, importlib.util, json, os, subprocess, sys, time, traceback
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE.parents[1] / 'configs/view_count.json'
C = json.loads(CONFIG_PATH.read_text())
sys.path.insert(0, str(HERE.parents[1] / 'src'))
from reconlab.benchmark.core import Geometry, save_json, sha256
from gaussian import GaussianBenchmark
from inputs import SourceBundle
from viewsets import settings
from pair_cache import CachedPairs
from filtering import FilterPolicy, FilteredMatcher
from reconlab.benchmark.matchers import RomaV2Matcher
from reconlab.benchmark.sfm import PairwiseReconstruction

OUT = Path(C['output'])


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def native():
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    free, total = torch.cuda.mem_get_info()
    assert free > 8 * 1024**3, 'Insufficient free GPU memory'
    torch.set_num_threads(4)
    pin = C['native_binary']
    assert sha256(Path(pin['path'])) == pin['sha256']
    spec = importlib.util.spec_from_file_location('gsplat_cuda', pin['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules['gsplat.csrc'] = module
    import gsplat
    gsplat.csrc = module
    from gsplat.cuda._backend import _C
    assert _C is module
    return {'name': torch.cuda.get_device_name(), 'free_bytes': free, 'total_bytes': total}


def cpu_tree(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


class MilestonePublisher:
    def __init__(self, item, geometry, evaluator, destination):
        self.item, self.geometry = item, geometry
        self.evaluator, self.destination = evaluator, destination

    def render_fixed_orbit(self, params, output):
        import torch
        from gsplat import rasterization
        metadata = json.loads(Path(self.item['fixed_orbit']).read_text())
        poses = align_orbit(metadata, json.loads((output/'pose_metrics.json').read_text()), *self.normalization)
        frames = []
        directory = output / 'centered'
        directory.mkdir()
        with torch.inference_mode():
            for i, pose in enumerate(poses):
                rgb, _, _ = rasterization(
                    means=params['means'], quats=params['quats'], scales=params['scales'].exp(),
                    opacities=params['opacities'].sigmoid(), colors=torch.cat([params['sh0'], params['shN']], 1),
                    viewmats=torch.tensor(np.linalg.inv(pose)[None], dtype=torch.float32, device='cuda'),
                    Ks=torch.tensor(np.array(metadata['intrinsics'])[None], dtype=torch.float32, device='cuda'),
                    width=512, height=512, sh_degree=2, backgrounds=torch.ones((1, 3), device='cuda'),
                    packed=False, near_plane=.01, far_plane=100)
                frame = Image.fromarray((rgb[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8))
                frame.save(directory / f'frame_{i:05d}.png')
                frames.append(frame)
        durations = [(round((i+1)*100/24)-round(i*100/24))*10 for i in range(72)]
        frames[0].save(output/'centered.gif', save_all=True, append_images=frames[1:], duration=durations, loop=0, disposal=2)
        save_json(output/'centered.json', {**metadata, 'fixed_across_checkpoints': True, 'render_camera_to_world_opencv':poses.tolist(),
                                          'source_sha256': sha256(Path(self.item['fixed_orbit']))})

    def __call__(self, step, params, optimizers, strategy, rng, order, center, scale, initial_points, elapsed):
        import torch
        output = self.destination / f'step_{step:06d}'
        output.mkdir()
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        checkpoint = {
            'state': cpu_tree(dict(params)), 'optimizers': cpu_tree({k: o.state_dict() for k, o in optimizers.items()}),
            'strategy_state': cpu_tree(strategy), 'numpy_rng_state': rng.bit_generator.state,
            'torch_rng_state': cpu_rng, 'cuda_rng_state': cuda_rng, 'view_order': order.copy(),
            'center': center, 'scene_scale': scale, 'steps': step, 'total_steps': C['steps'],
            'next_iteration_index': step, 'seed': 42, 'config_sha256': sha256(CONFIG_PATH)}
        temporary = output/'gaussians.pt.pending'
        torch.save(checkpoint, temporary)
        temporary.replace(output/'gaussians.pt')
        save_json(output/'training.json', {'steps': step, 'total_steps': C['steps'],
                  'initial_points': initial_points, 'final_points': len(params['means']),
                  'elapsed_including_previous_evaluations': elapsed, 'seed': 42,
                  'checkpoint_sha256': sha256(output/'gaussians.pt'),
                  'initialization': 'same filtered SfM point cloud; fresh optimizer',
                  'training_camera_source': 'estimated; fixed', 'lr_decay_horizon': 15000,
                  'refine_stop_iter': 2500, 'optimizer_and_rng_saved': True})
        del checkpoint
        save_json(self.destination/'status.json', {'state': 'RUNNING', 'stage': 'evaluation', 'step': step, 'checked_at': now()})
        try:
            self.normalization = (center, scale)
            self.evaluator.evaluate(self.geometry, Path(self.item['source']), output, params, center, scale)
            self.render_fixed_orbit(params, output)
        finally:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
        save_json(output/'COMPLETE.json', {'step': step, 'checked_at': now(), 'state': 'COMPLETE'})
        print(json.dumps({'uid': self.item['uid'], 'milestone_complete': step, 'time': now()}), flush=True)


def align_orbit(metadata, alignment, center, scene_scale):
    poses = np.array(metadata['camera_to_world_opencv'])
    rotation = np.array(alignment['rotation'])
    translation = np.array(alignment['translation'])
    scale = alignment['scale']
    assert scale > 1e-10 and scene_scale > 1e-10
    estimated = poses.copy()
    estimated[:, :3, :3] = np.einsum('ij,njk->nik', rotation.T, poses[:, :3, :3])
    estimated[:, :3, 3] = (((poses[:, :3, 3]-translation)@rotation/scale)-center)/scene_scale
    return estimated


def reuse_baseline(item, setting, destination):
    old = Path(item['baseline_cases'])
    geometry = Geometry.load(Path(item['baseline_geometry']))
    assert set(geometry.names) == set(setting.names)
    assert json.loads((old/'status.json').read_text())['state'] == 'COMPLETE'
    pins = {}
    for step in C['milestones']:
        source = old/f'step_{step:06d}'
        assert (source/'COMPLETE.json').exists()
        info = json.loads((source/'training.json').read_text())
        assert info['steps']==step and info['total_steps']==15000
        assert info['seed']==42 and info['lr_decay_horizon']==15000 and info['refine_stop_iter']==2500
        assert info['checkpoint_sha256']==sha256(source/'gaussians.pt')
        (destination/source.name).symlink_to(source)
        pins[str(source/'gaussians.pt')] = info['checkpoint_sha256']
    geometry.save(destination)
    for name in ['matching.json','visuals','filter_audit','model_inputs']:
        source = Path(item['baseline_geometry'])/name
        if source.exists(): (destination/name).symlink_to(source)
    (destination/'selection.json').symlink_to(old/'selection.json')
    save_json(destination/'baseline_reuse.json', {'source':str(old),'sha256':pins,
              'exact_same_6_images_and_15k_schedule':True,'timing':'reused, excluded from new inference timing'})
    save_json(destination/'status.json', {'state':'COMPLETE','reused':True,'step':15000,
              'input_views':6,'registered':6,'checked_at':now()})


def worker(shard):
    import torch
    gpu = native()
    save_json(OUT/'workers'/f'{shard}.json', {'state':'STARTED','gpu':gpu,'checked_at':now()})
    raw = None
    failures = []
    for uid in C['shards'][shard]:
        item = next(o for o in C['objects'] if o['uid']==uid)
        bundle = SourceBundle(item, OUT)
        for setting in settings(bundle.train_names):
            destination = OUT/'results'/uid/setting.key
            if destination.exists():
                raise RuntimeError('Existing case needs explicit recovery: '+str(destination))
            destination.mkdir(parents=True)
            images, evaluation = bundle.subset(setting,destination)
            save_json(destination/'status.json',{'state':'RUNNING','stage':'matching_sfm',
                      'input_views':setting.count,'sampling':setting.layout,'checked_at':now()})
            try:
                if setting.key=='uniform_06':
                    reuse_baseline(item,setting,destination)
                    continue
                if raw is None:
                    raw = RomaV2Matcher(Path(C['roma_source']))
                cache = CachedPairs(raw,bundle,OUT/'pair_cache'/uid,C['roma_checkpoint_sha256'])
                policy = FilterPolicy(True,C['mask']['score_minimum'],C['mask']['white_floor'],C['mask']['margin'])
                matcher = FilteredMatcher(cache,policy)
                start = time.monotonic()
                geometry = PairwiseReconstruction(matcher,'romav2_fg_conf050').reconstruct(images,destination)
                geometry.metadata['reconstruction_seconds']=time.monotonic()-start
                geometry.save(destination)
                save_json(destination/'status.json',{'state':'RUNNING','stage':'training','step':0,
                          'input_views':setting.count,'registered':len(geometry.names),'checked_at':now()})
                gaussian = GaussianBenchmark(C['steps'])
                evaluation_item = {**item,'source':str(evaluation),'fixed_orbit':str(bundle.canonical_orbit)}
                gaussian.on_checkpoint = MilestonePublisher(evaluation_item,geometry,gaussian,destination)
                params,_,_ = gaussian.fit(geometry,images,destination/'training')
                vals = []
                for step in C['milestones']:
                    rows=json.loads((destination/f'step_{step:06d}/render_metrics.json').read_text())['views']
                    scores=[r['foreground_psnr'] for r in rows if r['split']=='val']
                    assert len(scores)==6 and all(x is not None and np.isfinite(x) for x in scores)
                    vals.append((float(np.mean(scores)),step))
                best=max(vals,key=lambda v:(v[0],-v[1]))
                save_json(destination/'selection.json',{'best_step':best[1],'val_foreground_psnr':best[0],'test_used':False})
                del params, gaussian
                save_json(destination/'status.json',{'state':'COMPLETE','step':15000,'input_views':setting.count,
                          'registered':len(geometry.names),'checked_at':now()})
            except Exception as error:
                (destination/'error.log').write_text(traceback.format_exc())
                save_json(destination/'status.json',{'state':'FAILED','error':repr(error),'input_views':setting.count,'checked_at':now()})
                failures.append({'uid':uid,'setting':setting.key,'error':repr(error)})
                if not any(reason in str(error) for reason in ['NO_RECONSTRUCTION','Insufficient reconstructed geometry','No usable initialization points','Degenerate camera scale']):
                    raise
            finally:
                gc.collect()
                torch.cuda.empty_cache()
    save_json(OUT/'workers'/f'{shard}.json',{'state':'COMPLETE','sfm_failures':failures,'checked_at':now()})


def main():
    os.umask(0o022)
    if len(sys.argv) > 1 and sys.argv[1] == 'worker':
        worker(int(sys.argv[2]))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'logs').mkdir(exist_ok=True)
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.free', '--format=csv,noheader'], text=True)
    assert len(inventory.strip().splitlines()) == 4
    save_json(OUT/'resource_check.json', {'inventory': inventory, 'checked_at': now()})
    children = []
    for shard in range(4):
        stream = (OUT/'logs'/f'worker_{shard}.log').open('w')
        process = subprocess.Popen([C['python'], '-u', str(HERE/'runtime.py'), 'worker', str(shard)],
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard)), stdout=stream, stderr=subprocess.STDOUT)
        children.append((process, stream))
    while any(p.poll() is None for p, _ in children):
        save_json(OUT/'heartbeat.json', {'checked_at': now(), 'workers': [p.poll() for p, _ in children]})
        time.sleep(20)
    codes = [p.wait() for p, _ in children]
    for _, stream in children:
        stream.close()
    save_json(OUT/'execution_complete.json', {'state': 'COMPLETE' if not any(codes) else 'FAILED', 'codes': codes, 'checked_at': now()})
    if any(codes):
        raise RuntimeError('Worker failure: '+str(codes))


if __name__ == '__main__':
    main()
