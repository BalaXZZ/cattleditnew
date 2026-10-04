"""Create a provisional identity manifest from breed folders and filename numbers."""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'raw'
OUT = ROOT / 'outputs'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def main():
    rows, unmatched = [], []
    hashes = defaultdict(list)
    for path in sorted(DATA.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        relative = path.relative_to(DATA)
        match = re.search(r'\d+', path.stem)
        if len(relative.parts) < 2 or not match:
            unmatched.append(relative.as_posix())
            continue
        breed = relative.parts[0]
        prefix = re.sub(r'[^a-z0-9]+', '_', breed.lower()).strip('_')
        animal_id = f'{prefix}_cow_{int(match.group()):04d}'
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        row = {'image_path': relative.as_posix(), 'animal_id': animal_id,
               'breed': breed, 'session_id': '', 'sha256': digest.hexdigest()}
        rows.append(row)
        hashes[row['sha256']].append(row)
    counts = Counter(row['animal_id'] for row in rows)
    conflicts = [group for group in hashes.values() if len({r['animal_id'] for r in group}) > 1]
    report = {'animal_count': len(counts), 'image_count': len(rows),
              'images_per_animal': dict(sorted(counts.items())),
              'animals_with_fewer_than_3_images': {k: v for k, v in sorted(counts.items()) if v < 3},
              'cross_identity_duplicate_groups': conflicts, 'unmatched_files': unmatched,
              'label_rule': 'Provisional identity = breed folder + first number in filename; user confirmed numbering identifies cows. Review exceptions and duplicates before training.',
              'sessions': 'Unknown; image suffixes are not treated as separate sessions.'}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'identity_manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image_path', 'animal_id', 'breed', 'session_id', 'sha256'])
        writer.writeheader()
        writer.writerows(rows)
    (OUT / 'identity_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Provisional cows: {len(counts)} | Images: {len(rows)}')
    print(f'Cows with fewer than 3 images: {sum(v < 3 for v in counts.values())}')
    print(f'Cross-identity exact duplicate groups: {len(conflicts)}')
    print(f'Unmatched files: {len(unmatched)}')


if __name__ == '__main__':
    main()
