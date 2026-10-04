"""Select muzzle-sized proposals for review; not automatic acceptance."""
import json
from pathlib import Path
from PIL import Image,ImageOps
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/auto_muzzles';DATA=ROOT/'data/clean'
results=json.loads((OUT/'proposals.json').read_text());selected={}
for rel,r in results.items():
    if not r['candidates']:selected[rel]={'status':'not_found'};continue
    w,h=r['size']
    def rank(c):
        x1,y1,x2,y2=c['box'];fraction=(x2-x1)*(y2-y1)/(w*h)
        ratio=(x2-x1)/(y2-y1)
        return c['score']-0.8*max(0,fraction-0.2)
    candidate=max(r['candidates'],key=rank)
    box=candidate['box'];selected[rel]={'box':box,'score':candidate['score'],'status':'pending_visual_review'}
    with Image.open(DATA/rel) as im:
        crop=ImageOps.exif_transpose(im).convert('RGB').crop(box)
        path=OUT/'candidate_crops'/Path(rel).with_suffix('.png');path.parent.mkdir(parents=True,exist_ok=True);crop.save(path)
(OUT/'selected.json').write_text(json.dumps(selected,indent=2));print('Selected',len(selected),'proposals for visual review')
