"""Conservatively exclude unresolved IDs; prepare crop annotation template."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw'
CLEAN=ROOT/'data/clean'
OUT=ROOT/'outputs/image_audit'

def main():
    if CLEAN.exists(): raise RuntimeError('Clean dataset already exists; refusing overwrite.')
    report=json.loads((OUT/'audit_report.json').read_text(encoding='utf-8'))
    excluded=set(report['identity_conflicts'])|set(report['cows_with_fewer_than_3_unique_decoded_images'])
    with (OUT/'image_quality.csv').open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    accepted=[r for r in rows if r['animal_id'] not in excluded]
    CLEAN.mkdir()
    annotations=[]
    for r in accepted:
        src=RAW/r['image_path']; dst=CLEAN/r['image_path']
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        if hashlib.sha256(dst.read_bytes()).hexdigest()!=r['sha256']: raise RuntimeError(f'Copy mismatch: {dst}')
        annotations.append({'image_path':r['image_path'],'animal_id':r['animal_id'],
                            'width':r['width'],'height':r['height'],
                            'x_min':'','y_min':'','x_max':'','y_max':'',
                            'muzzle_usable':'pending','session_id':'','quality_flags':r['flags']})
    with (OUT/'muzzle_annotations.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(annotations[0]));writer.writeheader();writer.writerows(annotations)
    summary={'clean_cows':len({r['animal_id'] for r in accepted}),'clean_images':len(accepted),
             'excluded_ids':sorted(excluded),'excluded_images':len(rows)-len(accepted),
             'policy':'Exclude whole IDs with cross-identity pixel duplicates or fewer than 3 distinct images. Raw unchanged. IDs not renumbered.',
             'status':'Provisional clean copy; muzzle box annotation, visual quality review and identity confirmation remain necessary before training.'}
    (OUT/'clean_dataset_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
