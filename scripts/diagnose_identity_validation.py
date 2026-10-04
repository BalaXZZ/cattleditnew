"""Inspect only train/validation cows. Never load test photographs for tuning."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from train_muzzle_embeddings import CropDataset,encode
from src.biometrics import load_encoder,retrieval_protocol,summarize_protocol

def main():
    torch.set_num_threads(4);out=ROOT/'outputs/reliability';out.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'data/embedding/manifest.json').read_text());rows=[r for r in manifest['rows'] if r['split']=='val']
    model,_=load_encoder(ROOT/'outputs/embeddings/resnet18_arcface/best.pt','cuda')
    vectors=encode(model,DataLoader(CropDataset(rows),batch_size=16), 'cuda');np.savez(out/'baseline_val.npz',embeddings=vectors)
    result=retrieval_protocol(vectors,rows);(out/'baseline_validation.json').write_text(json.dumps(result,indent=2))
    lookup={c:[r for r in rows if r['animal_id']==c] for c in sorted({r['animal_id'] for r in rows},key=lambda c:int(c.split('_')[1]))}
    def paste(canvas,draw,rel,x,y,label):
        with Image.open(ROOT/'data/muzzle_crops'/rel) as source:
            im=source.convert('RGB');im.thumbnail((190,160));canvas.paste(im,(x,y))
        draw.text((x,y+163),label,fill='black')
    cows=list(lookup);canvas=Image.new('RGB',(1800,((len(cows)+2)//3)*185),'white');draw=ImageDraw.Draw(canvas)
    for j,c in enumerate(cows):
        for k,r in enumerate(lookup[c][:3]):paste(canvas,draw,r['image_path'],j%3*600+k*200,j//3*185,c+' '+Path(r['image_path']).stem[-8:])
    canvas.save(out/'validation_cows.jpg')
    failures=[r for r in result['closed_queries'] if not r['correct']]
    for start in range(0,len(failures),6):
        batch=failures[start:start+6];canvas=Image.new('RGB',(1000,len(batch)*195),'white');draw=ImageDraw.Draw(canvas)
        for j,f in enumerate(batch):
            items=[(f['source_path'],'query '+f['animal_id'])]
            for c,label in [(f['animal_id'],'true'),(f['predicted_id'],'pred')]:
                rr=lookup[c];n=len(rr);items.extend((rr[k]['image_path'],label+' '+c) for k in [f['fold']%n,(f['fold']+1)%n])
            for k,(rel,label) in enumerate(items):paste(canvas,draw,rel,k*200,j*195,label)
        canvas.save(out/f'validation_failures_{start//6+1:02d}.jpg')
    print(summarize_protocol(result));print('Validation failures:',len(failures))

if __name__=='__main__':main()
