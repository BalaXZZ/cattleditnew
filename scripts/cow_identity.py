"""Local prototype: full photo -> YOLO crop -> embedding -> reviewed cow candidate."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'work/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/matplotlib'))
import numpy as np
import torch
from src.biometrics import normalize
from src.registry import CowRegistry
from src.identity_system import IdentitySystem
from src.capture_pipeline import CapturePipeline

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--registry',type=Path,default=ROOT/'data/registry/cows.sqlite3')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--model-dir',type=Path,help='Directory containing best.pt and calibration.json; default uses selected active model')
    parser.add_argument('--cropped',action='store_true',help='Inputs are already muzzle crops; bypass YOLO')
    commands=parser.add_subparsers(dest='command',required=True)
    enroll=commands.add_parser('enroll');enroll.add_argument('--cow-id',required=True);enroll.add_argument('--details',type=Path);enroll.add_argument('--confirm-new',action='store_true');enroll.add_argument('images',type=Path,nargs='+')
    identify=commands.add_parser('identify');identify.add_argument('image',type=Path)
    commands.add_parser('list')
    args=parser.parse_args();torch.set_num_threads(4)
    active=ROOT/'outputs/embeddings/active_model.json'
    model_dir=args.model_dir
    if model_dir is None:
        model_dir=ROOT/json.loads(active.read_text())['model_dir'] if active.exists() else ROOT/'outputs/embeddings/resnet18_arcface'
    system=IdentitySystem(model_dir,args.device,load_models=args.command!='list')
    model_hash=system.model_hash;calibration=system.calibration;matching_mode=system.matching_mode
    registry=CowRegistry(args.registry,model_hash,dimension=system.dimension,matching_mode=matching_mode,branch_dimensions=system.dimensions,branch_weights=system.weights)
    try:
        if args.command=='list':print(json.dumps(registry.list_cows(),indent=2));return
        pipeline=CapturePipeline(system,ROOT/'outputs/yolo/muzzle_yolo11n/weights/best.pt')
        def embed(path):
            return pipeline.encode_path(path,cropped=args.cropped)
        if args.command=='identify':
            vector,_,box,quality=embed(args.image);candidates=registry.candidates(vector);n=registry.count()
            match=bool(candidates and candidates[0]['similarity']>=calibration['identification_threshold'])
            print(json.dumps({'status':'capture_review_required' if quality['flags'] else 'candidate_match_requires_review' if match else 'no_confident_match',
                              'muzzle_box':box,'candidates':candidates,'threshold':calibration['identification_threshold'],
                              'capture_quality':quality,
                              'registry_cows':n,'gallery_larger_than_calibration':n>calibration['validation']['known_gallery_cows'],
                              'model_sha256':model_hash,'matching_mode':matching_mode,'note':'No confident match does not prove a new cow. Independent accuracy has not been established for the current model; do not auto-register or certify a match.'},indent=2));return
        if len(args.images)<2:raise ValueError('Provide at least two distinct enrollment photos; three are preferable.')
        details=json.loads(args.details.read_text()) if args.details else {}
        results=[embed(path) for path in args.images];vectors=np.stack([r[0] for r in results]);prototype=normalize(vectors.mean(axis=0))
        candidates=registry.candidates(prototype)
        possible_duplicate=bool(candidates and candidates[0]['similarity']>=calibration['identification_threshold'])
        quality_flags=[{'image':str(path),**r[3]} for path,r in zip(args.images,results) if r[3]['flags']]
        pair_scores=[system.pair_score(vectors[i],vectors[j]) for i in range(len(vectors)) for j in range(i+1,len(vectors))]
        inconsistent=min(pair_scores)<calibration['pair_threshold']
        if (possible_duplicate or inconsistent or quality_flags) and not args.confirm_new:
            print(json.dumps({'status':'enrollment_review_required','possible_duplicate':possible_duplicate,'low_photo_consistency':inconsistent,
                              'candidates':candidates,'pair_similarities':pair_scores,'capture_quality_flags':quality_flags,'note':'Review photos and candidate IDs. Use --confirm-new only after confirming these photos belong to one new cow.'},indent=2));return
        registry.enroll(args.cow_id,details,vectors,[p.resolve() for p in args.images],[r[1] for r in results],input_kind='muzzle_crop' if args.cropped else 'full_image')
        print(json.dumps({'status':'enrolled','cow_id':args.cow_id,'photo_count':len(args.images),'registry':str(args.registry),'model_sha256':model_hash,'review_override':args.confirm_new},indent=2))
    finally:registry.close()

if __name__=='__main__':main()
