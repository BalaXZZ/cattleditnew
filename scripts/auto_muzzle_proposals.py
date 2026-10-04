"""Generate resumable muzzle proposals with Grounding DINO; not reviewed labels."""
import argparse
import csv
import json
import os
from pathlib import Path
import torch
from PIL import Image, ImageOps, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'models/detection_cache'
OUT=ROOT/'outputs/auto_muzzles'
DATA=ROOT/'data/clean'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=0);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print('Detector device:',device,flush=True)
    model_id='IDEA-Research/grounding-dino-tiny'
    processor=AutoProcessor.from_pretrained(model_id,cache_dir=CACHE)
    model=AutoModelForZeroShotObjectDetection.from_pretrained(ROOT/'models/grounding-dino-local',use_safetensors=True).to(device).eval()
    with (ROOT/'outputs/image_audit/muzzle_annotations.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    # Sample across breeds/identities before a full run.
    if args.limit: rows=rows[::max(1,len(rows)//args.limit)][:args.limit]
    target=OUT/'proposals.json'
    results=json.loads(target.read_text()) if target.exists() else {}
    for i,row in enumerate(rows):
        if (OUT/'STOP').exists():
            print('Stopped safely at image boundary.',flush=True);break
        rel=row['image_path']
        if rel in results:continue
        with Image.open(DATA/rel) as im:image=ImageOps.exif_transpose(im).convert('RGB')
        inputs=processor(images=image,text='the nose of a cow. the muzzle of a cow.',return_tensors='pt',size={'shortest_edge':384,'longest_edge':640}).to(device)
        with torch.inference_mode():outputs=model(**inputs)
        processed=processor.post_process_grounded_object_detection(outputs,inputs.input_ids,threshold=0.20,text_threshold=0.20,target_sizes=[image.size[::-1]])[0]
        candidates=[]
        w,h=image.size
        for score,box in zip(processed['scores'].tolist(),processed['boxes'].tolist()):
            x1,y1,x2,y2=box
            if not (x2>x1 and y2>y1):continue
            fraction=(x2-x1)*(y2-y1)/(w*h)
            if fraction>0.6 or fraction<0.002:continue
            candidates.append({'score':score,'box':[max(0,int(x1)),max(0,int(y1)),min(w,int(x2+1)),min(h,int(y2+1))]})
        candidates.sort(key=lambda c:c['score'],reverse=True)
        result={'animal_id':row['animal_id'],'candidates':candidates,'status':'proposal_only' if candidates else 'not_found','size':[w,h]}
        results[rel]=result
        temp=target.with_suffix('.tmp');temp.write_text(json.dumps(results,indent=2));os.replace(temp,target)
        if candidates:
            crop=image.crop(candidates[0]['box'])
            destination=OUT/'candidate_crops'/Path(rel).with_suffix('.png');destination.parent.mkdir(parents=True,exist_ok=True);crop.save(destination)
        print(f'{i+1}/{len(rows)} {rel}: {len(candidates)} candidates',flush=True)
    items=[(rel,r) for rel,r in results.items() if r['candidates']]
    for offset in range(0,len(items),48):
        batch=items[offset:offset+48];canvas=Image.new('RGB',(1200,((len(batch)+5)//6)*170),'white');draw=ImageDraw.Draw(canvas)
        for j,(rel,r) in enumerate(batch):
            with Image.open(OUT/'candidate_crops'/Path(rel).with_suffix('.png')) as im:
                thumb=im.copy();thumb.thumbnail((190,135));x=(j%6)*200;y=(j//6)*170;canvas.paste(thumb,(x,y));draw.text((x,y+136),f"{r['animal_id']} {r['candidates'][0]['score']:.2f}",fill='black');draw.text((x,y+151),Path(rel).name[:27],fill='black')
        canvas.save(OUT/f'crop_sheet_{offset//48+1:02d}.jpg')
    print('Proposals saved. Human/model review required; annotation CSV untouched.',flush=True)
if __name__=='__main__':main()
