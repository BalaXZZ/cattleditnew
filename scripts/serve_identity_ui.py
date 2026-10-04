"""Local browser UI for model-versioned cattle enrollment and identification."""
import argparse
import base64
import io
import json
import os
import sqlite3
import sys
import tempfile
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'work/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work/matplotlib'))
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
import torch
from src.identity_system import IdentitySystem
from src.capture_pipeline import CapturePipeline
from src.registry import CowRegistry
from src.biometrics import normalize

MAX_REQUEST = 40 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 24_000_000

class Application:
    def __init__(self, registry, device, model_dir=None):
        self.registry_path = Path(registry)
        self.device = device
        self.model_dir = model_dir or ROOT / json.loads((ROOT / 'outputs/embeddings/active_model.json').read_text())['model_dir']
        self.system = IdentitySystem(self.model_dir, device, load_models=False)
        self.pipeline = None
        self.lock = threading.Lock()

    def registry(self):
        s = self.system
        return CowRegistry(self.registry_path, s.model_hash, s.dimension, s.matching_mode, s.dimensions, s.weights)

    def load(self):
        if self.pipeline is None:
            self.system = IdentitySystem(self.model_dir, self.device)
            self.pipeline = CapturePipeline(self.system, ROOT / 'outputs/yolo/muzzle_yolo11n/weights/best.pt')

    @staticmethod
    def save_image(item, directory):
        try:
            raw = base64.b64decode(item['data'], validate=True)
            if not raw or len(raw) > 15 * 1024 * 1024:
                raise ValueError('Each image must be smaller than 15 MB.')
            with Image.open(io.BytesIO(raw)) as source:
                if source.width * source.height > 24_000_000:
                    raise ValueError('Image exceeds 24 megapixels. Resize it before uploading.')
                image = ImageOps.exif_transpose(source).convert('RGB')
                image.load()
        except (KeyError, UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError('Upload a valid JPEG, PNG or WebP image.') from exc
        path = directory / (uuid.uuid4().hex + '.png')
        image.save(path)
        return path

    @staticmethod
    def preview(path, box):
        with Image.open(path) as source:
            image = source.convert('RGB')
            if box is not None:
                image = image.crop(box)
            image.thumbnail((640, 480))
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=90)
        return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()

    def run(self, action, payload=None):
        payload = payload or {}
        with self.lock:
            registry = self.registry()
            try:
                if action == 'state':
                    cows = registry.list_cows()
                    for cow in cows:
                        cow['photos'] = [{'url': '/api/photo?cow_id=' + quote(cow['cow_id'], safe='') + '&index=' + str(i),
                                          'label': 'Photo ' + str(i+1)} for i in range(cow['photo_count'])]
                    return {'cows': cows, 'device': self.device, 'model': 'CNN + ViT',
                            'threshold': self.system.calibration['identification_threshold']}
                if action == 'upload':
                    directory = ROOT / 'work/ui_uploads'
                    directory.mkdir(parents=True, exist_ok=True)
                    path = self.save_image(payload, directory)
                    return {'token': path.stem}
                if action not in ('identify', 'enroll'):
                    raise ValueError('Unknown action.')
                cropped = payload.get('kind') == 'muzzle'
                if payload.get('kind') not in ('muzzle', 'full'):
                    raise ValueError('Select muzzle photo or full cow photo.')
                images = payload.get('images', [])
                if not isinstance(images, list) or not images:
                    raise ValueError('Select a photo.')
                if action == 'identify' and len(images) != 1:
                    raise ValueError('Choose exactly one identification photo.')
                self.load()
                (ROOT / 'work').mkdir(exist_ok=True)
                with tempfile.TemporaryDirectory(dir=ROOT / 'work', prefix='ui_capture_') as directory:
                    paths = []
                    for item in images:
                        if 'token' in item:
                            token = item['token']
                            if not isinstance(token, str) or len(token) != 32 or any(c not in '0123456789abcdef' for c in token):
                                raise ValueError('Invalid upload token.')
                            staged = ROOT / 'work/ui_uploads' / (token + '.png')
                            if not staged.is_file():
                                raise ValueError('Upload expired. Select the photos again.')
                            path = Path(directory) / (uuid.uuid4().hex + '.png')
                            path.write_bytes(staged.read_bytes())
                            paths.append(path)
                        else:
                            paths.append(self.save_image(item, Path(directory)))
                    results = [self.pipeline.encode_path(path, cropped=cropped) for path in paths]
                    if action == 'identify':
                        vector, digest, box, quality = results[0]
                        candidates = registry.candidates(vector)
                        match = bool(candidates and candidates[0]['similarity'] >= self.system.calibration['identification_threshold'])
                        status = 'registry_empty' if not candidates else 'capture_review_required' if quality['flags'] else 'candidate_match_requires_review' if match else 'no_confident_match'
                        # Do not reveal an unrelated registered cow for rejected captures.
                        accepted = status == 'candidate_match_requires_review'
                        return {'status': status, 'candidates': candidates[:1] if accepted else [], 'muzzle_box': box,
                                'muzzle_preview': self.preview(paths[0], box), 'quality': quality,
                                'threshold': self.system.calibration['identification_threshold'],
                                'registry_cows': registry.count(),
                                'gallery_larger_than_calibration': registry.count() > self.system.calibration['validation']['known_gallery_cows']}
                    cow_id = str(payload.get('cow_id', '')).strip()
                    details = payload.get('details', {})
                    if not cow_id or len(cow_id) > 128:
                        raise ValueError('Enter a cow ID of at most 128 characters.')
                    if not isinstance(details, dict):
                        raise ValueError('Invalid cow details.')
                    hashes = [r[1] for r in results]
                    if len(set(hashes)) != len(hashes):
                        raise ValueError('Use distinct photos, not repeated copies of one image.')
                    vectors = np.stack([r[0] for r in results])
                    candidates = registry.candidates(normalize(vectors.mean(axis=0)))
                    pair_scores = [self.system.pair_score(vectors[i], vectors[j]) for i in range(len(vectors)) for j in range(i+1, len(vectors))]
                    duplicate = bool(candidates and candidates[0]['similarity'] >= self.system.calibration['identification_threshold'])
                    inconsistent = bool(pair_scores) and min(pair_scores) < self.system.calibration['pair_threshold']
                    single_photo = len(paths) == 1
                    flags = [r[3]['flags'] for r in results]
                    review = duplicate or inconsistent or any(flags) or single_photo
                    if review and payload.get('confirm_review') is not True:
                        return {'status': 'enrollment_review_required', 'possible_duplicate': duplicate,
                                'low_photo_consistency': inconsistent, 'quality_flags': flags,
                                'single_photo': single_photo,
                                'candidates': candidates, 'pair_scores': pair_scores,
                                'previews': [self.preview(p, r[2]) for p, r in zip(paths, results)]}
                    # Preserve original normalized uploads for future model migrations.
                    destination = self.registry_path.parent / 'captures' / uuid.uuid4().hex
                    destination.mkdir(parents=True)
                    saved = []
                    try:
                        for path in paths:
                            target = destination / path.name
                            target.write_bytes(path.read_bytes())
                            saved.append(target.resolve())
                        registry.enroll(cow_id, details, vectors, saved, hashes, 'muzzle_crop' if cropped else 'full_image')
                    except Exception:
                        for path in saved:
                            path.unlink(missing_ok=True)
                        destination.rmdir()
                        raise
                    return {'status': 'enrolled', 'cow_id': cow_id, 'photo_count': len(saved),
                            'previews': [self.preview(p, r[2]) for p, r in zip(paths, results)]}
            finally:
                registry.close()
                if action in ('identify', 'enroll'):
                    for item in payload.get('images', []):
                        token = item.get('token') if isinstance(item, dict) else None
                        if isinstance(token, str) and len(token) == 32 and all(c in '0123456789abcdef' for c in token):
                            (ROOT / 'work/ui_uploads' / (token + '.png')).unlink(missing_ok=True)

    def photo(self, cow_id, index):
        with self.lock:
            registry = self.registry()
            try:
                sources = registry.connection.execute('SELECT source_path FROM templates WHERE cow_id=? ORDER BY rowid', (cow_id,)).fetchall()
                if index < 0 or index >= len(sources):
                    raise ValueError('Photo not found.')
                with Image.open(sources[index][0]) as source:
                    image = ImageOps.exif_transpose(source).convert('RGB')
                    image.thumbnail((640, 480))
                    buffer = io.BytesIO()
                    image.save(buffer, format='JPEG', quality=85)
                    return buffer.getvalue()
            finally:
                registry.close()

class Handler(BaseHTTPRequestHandler):
    def send(self, status, content, content_type='application/json'):
        data = json.dumps(content).encode() if content_type == 'application/json' else content
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            url = urlparse(self.path)
            if url.path == '/':
                self.send(200, (ROOT / 'ui/index.html').read_bytes(), 'text/html; charset=utf-8')
            elif url.path == '/api/state':
                self.send(200, self.server.application.run('state'))
            elif url.path == '/api/photo':
                query = parse_qs(url.query)
                self.send(200, self.server.application.photo(query['cow_id'][0], int(query['index'][0])), 'image/jpeg')
            else:
                self.send(404, {'error': 'Not found.'})
        except (ValueError, KeyError, FileNotFoundError) as exc:
            self.send(404, {'error': str(exc)})
        except Exception as exc:
            self.send(500, {'error': str(exc)})

    def do_POST(self):
        # Only the local UI may mutate this local registry; no cross-origin requests.
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            self.send(403, {'error': 'Cross-origin requests are not allowed.'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_REQUEST:
                self.send(413, {'error': 'Upload is too large (40 MB request limit).'})
                return
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                raise ValueError('Expected a JSON upload.')
            if self.path not in ('/api/identify', '/api/enroll', '/api/upload'):
                self.send(404, {'error': 'Not found.'})
                return
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('Invalid request.')
            self.send(200, self.server.application.run(self.path.rsplit('/', 1)[1], payload))
        except sqlite3.IntegrityError:
            self.send(400, {'error': 'This cow ID or one of these photos is already enrolled. Check the registry.'})
        except (ValueError, TypeError, KeyError) as exc:
            self.send(400, {'error': str(exc)})
        except Exception as exc:
            self.send(500, {'error': str(exc)})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--registry', type=Path, default=ROOT / 'data/registry/cows.sqlite3')
    parser.add_argument('--model-dir', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.application = Application(args.registry, args.device, args.model_dir)
    print(f'Cattle identity UI: http://127.0.0.1:{args.port}/ ({args.device})', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
