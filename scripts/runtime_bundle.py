"""Export/install the app's trained weights without photos or registry data."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'outputs/embeddings/active_model.json',
    *('outputs/embeddings/cnn_vit_v3/' + name for name in (
        'cnn.pt', 'vit.pt', 'system.json', 'calibration.json',
        'capture_policy.json', 'development_report.json')),
    'outputs/yolo/muzzle_yolo11n/weights/best.pt',
)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def export_bundle(path, root=ROOT):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Bundle already exists: {path}')
    contents = {name: (root / name).read_bytes() for name in FILES}
    active = json.loads(contents[FILES[0]])
    if active['model_dir'] != 'outputs/embeddings/cnn_vit_v3':
        raise ValueError('This exporter supports the current cnn_vit_v3 runtime only.')
    manifest = {'format_version': 1, 'sha256': {k: digest(v) for k, v in contents.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
        archive.writestr('manifest.json', json.dumps(manifest, indent=2))
    print(f'Runtime bundle: {path}\nSHA256: {digest(path.read_bytes())}')

def install_bundle(path, expected_sha256, root=ROOT):
    path = Path(path)
    if digest(path.read_bytes()) != expected_sha256.lower():
        raise ValueError('ZIP checksum does not match the checksum supplied by your teammate.')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(FILES) | {'manifest.json'}:
            raise ValueError('Unexpected, missing or duplicate files in runtime bundle.')
        if sum(info.file_size for info in archive.infolist()) > 500 * 1024 * 1024:
            raise ValueError('Runtime bundle is too large.')
        manifest = json.loads(archive.read('manifest.json'))
        if manifest.get('format_version') != 1 or set(manifest.get('sha256', {})) != set(FILES):
            raise ValueError('Invalid runtime manifest.')
        contents = {name: archive.read(name) for name in FILES}
    for name, data in contents.items():
        if digest(data) != manifest['sha256'][name]:
            raise ValueError(f'Corrupted runtime file: {name}')
        target = root / name
        if target.exists() and digest(target.read_bytes()) != digest(data):
            raise ValueError(f'Refusing to replace a different model: {target}')
    # Names are a fixed allowlist; never extract arbitrary archive paths.
    for name, data in contents.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print('Runtime installed. Your local enrolled-cow database was not changed.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['export', 'install'])
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--sha256', help='Required when installing; obtain from the bundle sender.')
    args = parser.parse_args()
    if args.action == 'export':
        export_bundle(args.bundle)
    else:
        if not args.sha256:
            parser.error('install requires --sha256')
        install_bundle(args.bundle, args.sha256)

if __name__ == '__main__':
    main()
