"""Export reviewed muzzle crops. Refuses pending annotations or overwrites."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
CSV=ROOT/'outputs/image_audit/muzzle_annotations.csv'
DATA=ROOT/'data/clean'
DEST=ROOT/'data/muzzle_crops'

def main():
    with CSV.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
    pending=sum(r['muzzle_usable']=='pending' for r in rows)
    if pending: raise SystemExit(f'Cannot export: {pending} images still need review.')
    if DEST.exists(): raise SystemExit('Crop folder exists; refusing overwrite.')
    accepted=[]
    for r in rows:
        if r['muzzle_usable']=='no':continue
        if r['muzzle_usable']!='yes':raise ValueError('Invalid review status')
        path=(DATA/r['image_path']).resolve()
        if not path.is_relative_to(DATA.resolve()):raise ValueError('Invalid path')
        with Image.open(path) as im:
            rgb=ImageOps.exif_transpose(im).convert('RGB');w,h=rgb.size
            box=tuple(int(r[k]) for k in ('x_min','y_min','x_max','y_max'))
            x1,y1,x2,y2=box
            if not(0<=x1<x2<=w and 0<=y1<y2<=h) or min(x2-x1,y2-y1)<32:raise ValueError(f'Invalid box: {path}')
        accepted.append((r,path,box))
    counts=Counter(r['animal_id'] for r,_,_ in accepted)
    eligible={cow for cow,count in counts.items() if count>=3}
    if not eligible:raise SystemExit('No cow has 3 accepted muzzle images.')
    DEST.mkdir()
    manifest=[]
    for r,path,box in accepted:
        if r['animal_id'] not in eligible:continue
        filename=Path(r['image_path']).with_suffix('.png')
        target=DEST/filename;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise ValueError('Crop filename collision')
        with Image.open(path) as im:ImageOps.exif_transpose(im).convert('RGB').crop(box).save(target)
        manifest.append({'animal_id':r['animal_id'],'image_path':filename.as_posix(),'source_path':r['image_path'],'session_id':r['session_id'],'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'box':list(box)})
    report={'cow_count':len(eligible),'crop_count':len(manifest),'excluded_after_review':{cow:n for cow,n in counts.items() if n<3},'images':manifest,'note':'Original-resolution RGB PNG crops. Resize and backbone normalization happen in the training loader; splits still required.'}
    (ROOT/'outputs/image_audit/crop_export.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(f'Exported {len(manifest)} crops from {len(eligible)} cows.')

if __name__=='__main__':main()
