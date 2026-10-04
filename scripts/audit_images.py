"""Non-destructive integrity/quality audit; visual sheets and duplicate evidence."""
import csv
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
OUT = ROOT / 'outputs/image_audit'
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def sheet(paths, filename, columns=6):
    w, h = 200, 180
    canvas = Image.new('RGB', (columns*w, ((len(paths)+columns-1)//columns)*h), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, p in enumerate(paths):
        x,y = (i%columns)*w, (i//columns)*h
        try:
            with Image.open(p) as im:
                thumb = ImageOps.exif_transpose(im).convert('RGB')
                thumb.thumbnail((190,140))
                canvas.paste(thumb, (x+(w-thumb.width)//2,y))
        except Exception:
            pass
        rel = p.relative_to(RAW)
        draw.text((x+4,y+142),rel.parts[0],fill='black')
        draw.text((x+4,y+156),p.name[:28],fill='black')
    canvas.save(filename)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths = sorted((p for p in RAW.rglob('*') if p.is_file() and p.suffix.lower() in EXT), key=lambda p:(int(p.parent.name.split('_')[1]),p.name.lower()))
    rows, hashes, pixel_hashes, dhashes, errors = [],defaultdict(list),defaultdict(list),[],[]
    for p in paths:
        rel = p.relative_to(RAW).as_posix()
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                im.load()
                rgb = ImageOps.exif_transpose(im).convert('RGB')
                width,height = rgb.size
                gray = np.asarray(rgb.convert('L').resize((224,224)),dtype=np.float32)
                lap = -4*gray[1:-1,1:-1]+gray[:-2,1:-1]+gray[2:,1:-1]+gray[1:-1,:-2]+gray[1:-1,2:]
                sharpness = float(lap.var())
                small = np.asarray(rgb.convert('L').resize((9,8)))
                bits = (small[:,1:]>small[:,:-1]).flatten()
                dh = sum(int(v)<<i for i,v in enumerate(bits))
                pixel = hashlib.sha256(f'{width}x{height}'.encode()+rgb.tobytes()).hexdigest()
                sha = hashlib.sha256(p.read_bytes()).hexdigest()
                flags=[]
                if min(width,height)<224: flags.append('small_image')
                if sharpness<30: flags.append('possible_blur')
                if gray.mean()<35: flags.append('dark')
                if gray.mean()>220: flags.append('bright')
                row={'image_path':rel,'animal_id':p.parent.name,'width':width,'height':height,'mean_gray':round(float(gray.mean()),2),'sharpness_proxy':round(sharpness,2),'flags':';'.join(flags),'sha256':sha,'pixel_sha256':pixel,'muzzle_review':'pending'}
                rows.append(row)
                hashes[sha].append(rel)
                pixel_hashes[pixel].append(rel)
                dhashes.append((rel,dh))
        except Exception as e:
            errors.append({'image_path':rel,'error':str(e)})
    exact=[v for v in hashes.values() if len(v)>1]
    decoded=[v for v in pixel_hashes.values() if len(v)>1]
    candidates=[]
    sha_by_path={r['image_path']:r['sha256'] for r in rows}
    for i,(a,ha) in enumerate(dhashes):
        for b,hb in dhashes[i+1:]:
            if sha_by_path[a]==sha_by_path[b]: continue
            dist=(ha^hb).bit_count()
            if dist<=4:
                candidates.append({'image_a':a,'image_b':b,'dhash_distance':dist,'cross_identity':a.split('/')[0]!=b.split('/')[0],'decision':'review_only'})
    conflicts=[v for v in decoded if len({p.split('/')[0] for p in v})>1]
    blocked={p.split('/')[0] for group in conflicts for p in group}
    unique_counts=Counter()
    for group in pixel_hashes.values():
        for cow in {p.split('/')[0] for p in group}: unique_counts[cow]+=1
    insufficient={cow:count for cow,count in unique_counts.items() if count<3}
    flagged=Counter(flag for r in rows for flag in r['flags'].split(';') if flag)
    report={'images_checked':len(paths),'valid_images':len(rows),'unreadable_images':errors,'exact_duplicate_groups':exact,'decoded_pixel_duplicate_groups':decoded,'cross_identity_duplicate_groups':conflicts,'identity_conflicts':sorted(blocked),'cows_with_fewer_than_3_unique_decoded_images':insufficient,'near_duplicate_candidates':len(candidates),'quality_flag_counts':dict(flagged),'quality_note':'Heuristic flags prioritize review; they do not prove blur, unusability or muzzle coverage. No raw images changed. Near duplicates are candidates, not confirmed duplicates.','dimensions':{'min_width':min(r['width'] for r in rows),'max_width':max(r['width'] for r in rows),'min_height':min(r['height'] for r in rows),'max_height':max(r['height'] for r in rows)}}
    write_csv(OUT/'image_quality.csv',rows,list(rows[0]))
    write_csv(OUT/'near_duplicate_candidates.csv',candidates,['image_a','image_b','dhash_distance','cross_identity','decision'])
    (OUT/'audit_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Every image is included in these sheets for human inspection.
    for start in range(0,len(paths),60):
        sheet(paths[start:start+60],OUT/f'all_images_{start//60+1:02d}.jpg')
    dup_paths=list(dict.fromkeys(p for g in decoded for p in g))
    if dup_paths: sheet([RAW/p for p in dup_paths],OUT/'exact_duplicates.jpg',columns=4)
    # First image for each cow, sampled across the ordered dataset.
    first={}
    for p in paths: first.setdefault(p.parent.name,p)
    representatives=list(first.values())[::7]
    sheet(representatives,OUT/'representative_sample.jpg')
    print(json.dumps({k:v for k,v in report.items() if k not in {'exact_duplicate_groups','decoded_pixel_duplicate_groups','cross_identity_duplicate_groups'}},indent=2))

if __name__=='__main__': main()
