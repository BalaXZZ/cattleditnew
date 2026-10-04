"""Reuse detector identity splits; never regroup photos randomly."""
import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    detector=json.loads((ROOT/'data/yolo_muzzle/manifest.json').read_text())
    assignments={r['animal_id']:r['split'] for r in detector['records']}
    crops=json.loads((ROOT/'outputs/image_audit/crop_export.json').read_text())
    rows=sorted(crops['images'],key=lambda r:(int(r['animal_id'].split('_')[1]),r['image_path']))
    for row in rows:
        row['split']=assignments[row['animal_id']]
        if not (ROOT/'data/muzzle_crops'/row['image_path']).is_file():raise ValueError('Missing crop')
    classes=sorted({r['animal_id'] for r in rows if r['split']=='train'},key=lambda c:int(c.split('_')[1]))
    counts=Counter(r['animal_id'] for r in rows)
    assert min(counts.values())>=3
    sets={s:{r['animal_id'] for r in rows if r['split']==s} for s in ('train','val','test')}
    assert not(sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test'])
    out=ROOT/'data/embedding';out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():raise SystemExit('Manifest exists; refusing overwrite.')
    report={'seed':42,'train_classes':classes,'split_images':dict(Counter(r['split'] for r in rows)),
            'split_cows':{s:len(v) for s,v in sets.items()},'rows':rows,
            'protocol':'Three fixed folds, two real enrollment photos per cow, remaining real photos as queries. No query augmentation. Capture sessions unknown; repeated folds are dependent.'}
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k not in ('rows','train_classes')})

if __name__=='__main__':main()
