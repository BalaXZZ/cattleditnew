"""Publish the validation-selected local model while preserving the baseline."""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.capture_quality import capture_metrics

def main():
    base=ROOT/'outputs/reliability/experiments';selection=json.loads((base/'selection.json').read_text());winner=selection['winner']
    if winner['name']=='baseline':raise SystemExit('No validated improvement; keep baseline active.')
    source=base/winner['name'];destination=ROOT/'outputs/embeddings/reliability_v2'
    if destination.exists():raise SystemExit('Reliability model directory exists; refusing overwrite.')
    destination.mkdir();shutil.copy2(source/'best.pt',destination/'best.pt');shutil.copy2(source/'calibration.json',destination/'calibration.json')
    manifest=json.loads((ROOT/'data/embedding/manifest.json').read_text());rows=[r for r in manifest['rows'] if r['split']=='train'];metrics=[]
    for r in rows:
        with Image.open(ROOT/'data/muzzle_crops'/r['image_path']) as im:metrics.append(capture_metrics(im))
    policy={'min_short_side':96,'sharpness_review_below':float(np.percentile([m['sharpness'] for m in metrics],2)),
            'contrast_review_below':20.0,'training_images':len(rows),'note':'Heuristic flags only, not a validated quality classifier. No validation/test examples were dropped from reported identification scores.'}
    (destination/'capture_policy.json').write_text(json.dumps(policy,indent=2))
    report={'selection':selection,'test_evaluation':'Not performed this round. Existing baseline test remains historical; fresh independent field photos required.',
            'model_sha256':hashlib.sha256((destination/'best.pt').read_bytes()).hexdigest(),'capture_policy':policy}
    (destination/'development_report.json').write_text(json.dumps(report,indent=2))
    active=ROOT/'outputs/embeddings/active_model.json'
    if active.exists():shutil.copy2(active,active.with_name('active_model_before_v2.json'))
    temp=active.with_suffix('.tmp');temp.write_text(json.dumps({'model_dir':destination.relative_to(ROOT).as_posix(),'model_sha256':report['model_sha256'],'status':'development_only_requires_review'},indent=2));os.replace(temp,active)
    print(json.dumps({'active_model':str(destination),'validation':winner['validation'],'capture_policy':policy},indent=2))

if __name__=='__main__':main()
