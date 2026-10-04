"""Export full EXIF-oriented images and reviewed muzzle boxes for detection."""
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/yolo_muzzle'

def main():
    if DEST.exists():
        raise SystemExit('Dataset already exists; refusing overwrite.')
    with (ROOT/'outputs/image_audit/muzzle_annotations.csv').open(newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['muzzle_usable'] == 'yes']
    cows = sorted({r['animal_id'] for r in rows}, key=lambda x: int(x.split('_')[1]))
    random.Random(42).shuffle(cows)
    nval = max(1, round(len(cows)*0.15)); ntest = max(1, round(len(cows)*0.15))
    assignments = {c: ('val' if i<nval else 'test' if i<nval+ntest else 'train') for i,c in enumerate(cows)}
    records=[]; hashes={}
    for r in rows:
        source=ROOT/'data/clean'/r['image_path']; split=assignments[r['animal_id']]
        with Image.open(source) as im:
            im=ImageOps.exif_transpose(im).convert('RGB'); w,h=im.size
            x1,y1,x2,y2=[int(r[k]) for k in ('x_min','y_min','x_max','y_max')]
            assert 0<=x1<x2<=w and 0<=y1<y2<=h, source
            digest=hashlib.sha256(im.tobytes()).hexdigest()
            if digest in hashes: raise ValueError(f'Duplicate image: {source} and {hashes[digest]}')
            hashes[digest]=r['image_path']
            stem=r['animal_id']+'__'+Path(r['image_path']).stem
            target=DEST/'images'/split/(stem+'.jpg'); label=DEST/'labels'/split/(stem+'.txt')
            target.parent.mkdir(parents=True, exist_ok=True); label.parent.mkdir(parents=True, exist_ok=True)
            if target.exists(): raise ValueError('Filename collision')
            im.save(target, quality=95, subsampling=0)
        coords=((x1+x2)/(2*w), (y1+y2)/(2*h), (x2-x1)/w, (y2-y1)/h)
        label.write_text('0 '+' '.join(f'{v:.8f}' for v in coords)+'\n')
        records.append({'animal_id':r['animal_id'], 'source_path':r['image_path'], 'split':split, 'image':target.relative_to(DEST).as_posix(), 'box':[x1,y1,x2,y2], 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
    (DEST/'dataset.yaml').write_text(f'path: {json.dumps(DEST.as_posix())}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: muzzle\n')
    summary={'seed':42, 'images':dict(Counter(r['split'] for r in records)), 'cows':dict(Counter(assignments.values())), 'records':records, 'note':'Cow-disjoint detection splits. All accepted boxes used, including cows with fewer than 3 captures. Rejected blurry photos are omitted, not labeled as background. Labels are assistant-reviewed pseudo-labels; test metrics measure agreement with these labels, not field accuracy. Keep these cow assignments for subsequent identity evaluation to avoid detector-training leakage.'}
    (DEST/'manifest.json').write_text(json.dumps(summary,indent=2))
    print({k:v for k,v in summary.items() if k!='records'})

if __name__=='__main__': main()
