"""Stage cow folders using complete filename prefixes; retain >=3 photos."""
import csv
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
STAGE = ROOT / 'data/raw_prepared'
OUT = ROOT / 'outputs'
EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def natural(value):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', value.lower())]

def main():
    if STAGE.exists():
        raise RuntimeError('Staging folder already exists; refusing overwrite.')
    groups = defaultdict(list)
    for p in sorted(RAW.rglob('*')):
        if p.is_file() and p.suffix.lower() in EXT:
            rel = p.relative_to(RAW)
            if len(rel.parts) != 2:
                raise RuntimeError(f'Expected breed/image layout: {rel}')
            # Remove trailing _a or _(a), or (a) before descriptive text.
            prefix = re.sub(r'[_ ]?\([a-z]\)(?=_|$)', '', p.stem, flags=re.I)
            prefix = re.sub(r'_[a-z]$', '', prefix, flags=re.I)
            groups[(rel.parts[0], prefix.casefold())].append(p)
    retained = [(k, v) for k, v in groups.items() if len(v) >= 3]
    retained.sort(key=lambda item: (natural(item[0][0]), natural(item[0][1])))
    excluded = [{'breed': k[0], 'prefix': k[1], 'count': len(v),
                 'files': [p.relative_to(RAW).as_posix() for p in v]}
                for k, v in groups.items() if len(v) < 3]
    STAGE.mkdir()
    rows, hashes = [], defaultdict(list)
    for i, ((breed, prefix), files) in enumerate(retained, 1):
        cow = f'cow_{i}'
        folder = STAGE / cow
        folder.mkdir()
        for p in files:
            target = folder / p.name
            shutil.copy2(p, target)
            h = digest(p)
            if digest(target) != h:
                raise RuntimeError(f'Copy verification failed: {p}')
            row = {'animal_id': cow, 'image_path': f'{cow}/{p.name}',
                   'breed': breed, 'source_prefix': prefix,
                   'original_path': p.relative_to(RAW).as_posix(), 'sha256': h}
            rows.append(row)
            hashes[h].append(row)
    duplicates = [v for v in hashes.values() if len(v) > 1]
    report = {'kept_cows': len(retained), 'kept_images': len(rows),
              'excluded_cows': len(excluded), 'excluded_images': sum(x['count'] for x in excluded),
              'excluded_groups': excluded,
              'groups_with_more_than_3_images': sum(len(v) > 3 for _, v in retained),
              'duplicate_groups': duplicates,
              'policy': 'Keep all photos for prefixes with at least 3 files. Do not deduplicate or merge different prefixes. Originals backed up outside raw.'}
    OUT.mkdir(exist_ok=True)
    with (OUT / 'cow_mapping.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'preprocessing_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in {'excluded_groups', 'duplicate_groups'}}, indent=2))

if __name__ == '__main__':
    main()
