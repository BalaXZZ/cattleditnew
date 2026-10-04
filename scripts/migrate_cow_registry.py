"""Re-embed into a NEW registry, preserving IDs/details and the original database."""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.identity_system import IdentitySystem
from src.capture_pipeline import CapturePipeline
from src.registry import CowRegistry
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'work/ultralytics'));os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/matplotlib'))

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('destination',type=Path);p.add_argument('--model-dir',type=Path);p.add_argument('--device',default='cpu');args=p.parse_args()
    torch.set_num_threads(4)
    if not args.source.is_file():raise SystemExit('Source registry missing')
    if args.destination.exists():raise SystemExit('Destination exists; refusing overwrite')
    staging=args.destination.with_name(args.destination.name+'.staging')
    if staging.exists():raise SystemExit('Existing migration staging file needs review')
    directory=args.model_dir or ROOT/json.loads((ROOT/'outputs/embeddings/active_model.json').read_text())['model_dir']
    # Read-only snapshot of the current registry. Original records are never edited.
    connection=sqlite3.connect(args.source.resolve().as_uri()+'?mode=ro',uri=True)
    try:
        connection.execute('BEGIN');cows=list(connection.execute('SELECT cow_id,details FROM cows ORDER BY cow_id'))
        photos=list(connection.execute('SELECT cow_id,image_hash,source_path,input_kind FROM templates'))
    finally:connection.close()
    for _,_,source,kind in photos:
        if kind not in ('muzzle_crop','full_image'):raise ValueError('Unrecorded input type; review source records before migration.')
        if not Path(source).is_file():raise ValueError(f'Source photo unavailable: {source}')
    system=IdentitySystem(directory,args.device);pipeline=CapturePipeline(system,ROOT/'outputs/yolo/muzzle_yolo11n/weights/best.pt')
    registry=CowRegistry(staging,system.model_hash,system.dimension,system.matching_mode,system.dimensions,system.weights);review=[]
    try:
        for cow,details in cows:
            items=[r for r in photos if r[0]==cow];results=[pipeline.encode_path(Path(r[2]),r[3]=='muzzle_crop') for r in items]
            # Detect modified source photos rather than silently changing provenance.
            for old,new in zip(items,results):
                if old[1]!=new[1]:raise ValueError('Source photo pixels changed since enrollment')
            kinds={r[3] for r in items}
            if len(kinds)!=1:raise ValueError('Mixed input types need explicit migration review')
            import numpy as np
            registry.enroll(cow,json.loads(details),np.stack([r[0] for r in results]),[r[2] for r in items],[r[1] for r in results],input_kind=items[0][3])
            review.extend({'cow_id':cow,'source_path':old[2],'flags':new[3]['flags']} for old,new in zip(items,results) if new[3]['flags'])
        if registry.count()!=len(cows):raise ValueError('Migration count mismatch')
        report={'source':str(args.source.resolve()),'destination':str(args.destination.resolve()),'cow_count':len(cows),'template_count':len(photos),'model_hash':system.model_hash,'quality_review':review,'note':'IDs and details preserved; templates re-encoded. Review quality flags before using the new registry.'}
    finally:registry.close()
    # Exclusive publication prevents overwriting a destination created meanwhile.
    with staging.open('rb') as src,args.destination.open('xb') as dst:
        import shutil
        shutil.copyfileobj(src,dst)
    args.destination.with_suffix('.migration.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
