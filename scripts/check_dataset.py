"""Inventory identity folders and detect byte-identical image files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data' / 'raw')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs')
    args = parser.parse_args()
    data = args.data_dir.resolve()
    if not data.is_dir():
        parser.error(f'Dataset directory does not exist: {data}')
    rows, counts, hashes, warnings = [], {}, {}, []
    for animal in sorted(p for p in data.iterdir() if p.is_dir()):
        images = sorted(p for p in animal.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
        counts[animal.name] = len(images)
        if len(images) < 3:
            warnings.append(f'{animal.name}: only {len(images)} images; expected at least 3.')
        for path in images:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            relative = path.relative_to(data).as_posix()
            value = digest.hexdigest()
            hashes.setdefault(value, []).append(relative)
            rows.append({'image_path': relative, 'animal_id': animal.name,
                         'session_id': '', 'breed': '', 'sha256': value})
    loose = sorted(p.name for p in data.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if loose:
        warnings.append('Images outside animal folders are excluded: ' + ', '.join(loose))
    duplicates = [paths for paths in hashes.values() if len(paths) > 1]
    if duplicates:
        warnings.append('Exact duplicate images found; inspect them before splitting data.')
    if not rows:
        warnings.append('No images found. Add individual-cow folders under data/raw.')
    report = {'identity_folder_count': len(counts), 'image_count': len(rows),
              'images_per_animal': counts, 'exact_duplicate_groups': duplicates,
              'warnings': warnings,
              'limitations': 'File inventory only; image decoding, muzzle quality, near duplicates and identity labels are not validated.'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'dataset_manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image_path', 'animal_id', 'session_id', 'breed', 'sha256'])
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / 'dataset_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Candidate identity folders: {len(counts)} | Images: {len(rows)}')
    for warning in warnings:
        print(f'WARNING: {warning}')
    print(f'Reports saved to: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
