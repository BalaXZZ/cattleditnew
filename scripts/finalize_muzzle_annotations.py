"""Apply completed visual decisions to annotation CSV, preserving originals."""
import csv
import json
import math
import os
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/auto_muzzles';CSV=ROOT/'outputs/image_audit/muzzle_annotations.csv'

def main():
    proposals=json.loads((OUT/'proposals.json').read_text());selected=json.loads((OUT/'selected.json').read_text());decisions=json.loads((OUT/'review_decisions.json').read_text())
    with CSV.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    missing=[r['image_path'] for r in rows if r['image_path'] not in decisions]
    if missing:raise SystemExit(f'{len(missing)} images still need visual review.')
    for row in rows:
        rel=row['image_path'];decision=decisions[rel];status=decision['status']
        if status not in {'yes','no'}:raise ValueError('Invalid review decision')
        row['muzzle_usable']=status;row['reviewer']='assistant visual review';row['review_reason']=decision['reason']
        if status=='no':
            for k in ('x_min','y_min','x_max','y_max'):row[k]=''
            continue
        if 'box' in decision:box=decision['box']
        elif 'candidate_index' in decision:box=proposals[rel]['candidates'][decision['candidate_index']]['box']
        else:box=selected[rel]['box']
        w,h=proposals[rel]['size'];x1,y1,x2,y2=box
        if not(0<=x1<x2<=w and 0<=y1<y2<=h):raise ValueError(f'Invalid box {rel}')
        # Preserve small context around detector box to avoid clipping muzzle edges.
        mx=math.ceil((x2-x1)*0.04);my=math.ceil((y2-y1)*0.04)
        box=[max(0,x1-mx),max(0,y1-my),min(w,x2+mx),min(h,y2+my)]
        for k,v in zip(('x_min','y_min','x_max','y_max'),box):row[k]=str(v)
    backup=CSV.with_name('muzzle_annotations_before_auto.csv')
    if not backup.exists():shutil.copy2(CSV,backup)
    temp=CSV.with_suffix('.tmp')
    with temp.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.replace(temp,CSV)
    summary={'reviewed_images':len(rows),'accepted_images':sum(r['muzzle_usable']=='yes' for r in rows),'rejected_images':sum(r['muzzle_usable']=='no' for r in rows),'method':'Grounding DINO proposals + assistant contact-sheet/source review, 4% box margin. Not a claim of pixel-perfect segmentation or guaranteed recognition accuracy.'}
    (OUT/'annotation_summary.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':main()
