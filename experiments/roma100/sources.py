"""Generate only missing source sets with the already approved frozen CUDA policy."""
from pathlib import Path
import json,os,subprocess
from filtering import BASE
from reconlab.benchmark.core import sha256,save_json

def ensure_source(uid,config,out):
    item=next(o for o in config['objects'] if o['uid']==uid)
    if sha256(Path(item['glb_path']))!=item['glb_sha256']:raise ValueError('GLB hash changed')
    previous=Path(config['source_root'])/'source_inputs'/uid
    if (previous/'INPUTS_COMPLETE.json').exists():return previous
    src=out/'source_inputs'/uid
    if (src/'INPUTS_COMPLETE.json').exists():return src
    if src.exists():raise RuntimeError('Partial source needs review: '+str(src))
    catalog=out/'catalogs'/f'{uid}.json'
    save_json(catalog,{'objects':[{'id':uid,'local_path':item['glb_path'],'uid':uid,'name':item['name'],'license':item['license'],'url':item['url']}]})
    env=dict(os.environ,PYTHONPATH=str(BASE/'src'))
    log=out/'logs'/f'source_{uid}.log';log.parent.mkdir(exist_ok=True)
    with log.open('w') as stream:
        subprocess.run([config['render_python'],'-m','reconlab.backends.blender_source','--catalog',str(catalog),'--output',str(src.parent),'--threads','2','--device','CUDA','--stage','inputs'],env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=3600,check=True)
    evidence=json.loads((src/'inputs.json').read_text())
    if evidence['source']['glb_sha256']!=item['glb_sha256'] or evidence['source'].get('cycles_backend')!='CUDA':raise ValueError('Source protocol mismatch')
    if evidence['policy']!=config['source_policy']:raise ValueError('Source policy mismatch')
    if not (src/'INPUTS_COMPLETE.json').exists():raise ValueError('Missing source completion marker')
    return src
