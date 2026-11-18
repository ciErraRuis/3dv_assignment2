"""CPU comparison using identical saved RoMa predictions and fresh geometry fits."""
from pathlib import Path
import json,sys,traceback,datetime,argparse,os
from filtering import FilterPolicy,FilteredMatcher,FrozenPairs,BASE
from reconlab.benchmark.sfm import PairwiseReconstruction,IncrementalMapper
from reconlab.benchmark.core import save_json
from reconlab.benchmark.visuals import export_geometry
OLD=Path('/mnt/afs/260010098/data/3dv_assignment/matching10_sparse6_r01')
OUT=Path('/mnt/afs/260010098/data/3dv_assignment/roma100_filter_r01')

def run(uid,variant):
    output=OUT/'comparison_geometry'/variant/uid;output.mkdir(parents=True,exist_ok=True)
    status=output/'CPU_STATUS.json'
    if status.exists():return json.loads(status.read_text())
    save_json(status,{'state':'RUNNING','uid':uid,'variant':variant})
    try:
        matcher=FilteredMatcher(FrozenPairs(OLD,uid),FilterPolicy(minimum_overlap=.5 if variant=='fg_conf050' else 0))
        geometry=PairwiseReconstruction(matcher,variant,IncrementalMapper(2)).reconstruct(OLD/'inputs'/uid/'images',output)
        geometry.save(output);export_geometry(geometry,OLD/'source_inputs'/uid,output)
        result={'state':'COMPLETE','uid':uid,'variant':variant,'registered':len(geometry.names),'points':len(geometry.points),'time':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    except Exception as e:
        result={'state':'FAILED','uid':uid,'variant':variant,'error':repr(e),'traceback':traceback.format_exc()}
    save_json(status,result);print(json.dumps(result),flush=True);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('uid');p.add_argument('variant',choices=['fg','fg_conf050']);a=p.parse_args();run(a.uid,a.variant)
