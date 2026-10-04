"""Exercise the UI backend with real models and an isolated, temporary registry."""
import base64
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from serve_identity_ui import Application, ROOT
from src.registry import CowRegistry

def main():
    import torch
    torch.set_num_threads(4)
    def item(path):
        return {'data': base64.b64encode(path.read_bytes()).decode()}
    crops = sorted((ROOT / 'data/muzzle_crops/cow_8').glob('*.png'))
    full = ROOT / 'data/clean/cow_8/Brindawan_8_c.jpeg'
    assert len(crops) >= 3 and full.exists()
    checks = []
    with tempfile.TemporaryDirectory(dir=ROOT / 'work', prefix='ui_test_') as directory:
        app = Application(Path(directory) / 'cows.sqlite3', 'cuda' if torch.cuda.is_available() else 'cpu')
        assert app.run('state')['cows'] == []
        checks.append('empty_registry')
        empty = app.run('identify', {'kind': 'muzzle', 'images': [item(crops[0])]})
        assert empty['status'] == 'registry_empty' and empty['muzzle_preview'].startswith('data:image/jpeg;base64,')
        checks.append('cropped_input_and_preview')
        payload = {'kind': 'muzzle', 'images': [item(crops[0]), item(crops[1])], 'cow_id': 'TEST-COW-8', 'details': {'owner': 'UI test', 'breed': 'Brindawan'}}
        result = app.run('enroll', payload)
        if result['status'] == 'enrollment_review_required':
            assert not app.run('state')['cows']
            payload['confirm_review'] = True
            result = app.run('enroll', payload)
        assert result['status'] == 'enrolled'
        assert app.run('state')['cows'][0]['photo_count'] == 2
        checks.append('reviewed_enrollment_and_details')
        lookup = app.run('identify', {'kind': 'muzzle', 'images': [item(crops[0])]})
        if lookup['status'] == 'candidate_match_requires_review':
            assert lookup['candidates'][0]['cow_id'] == 'TEST-COW-8'
            assert lookup['candidates'][0]['details']['owner'] == 'UI test'
        else:
            assert lookup['candidates'] == []
        checks.append('enrolled_identity_lookup')
        full_lookup = app.run('identify', {'kind': 'full', 'images': [item(full)]})
        assert len(full_lookup['muzzle_box']) == 4
        if full_lookup['status'] != 'candidate_match_requires_review':
            assert full_lookup['candidates'] == []
        checks.append('full_photo_yolo_crop')
        try:
            app.run('enroll', {**payload, 'cow_id': 'DUPLICATE', 'images': [item(crops[0]), item(crops[0])]})
            raise AssertionError('Duplicate image accepted')
        except ValueError as exc:
            assert 'distinct' in str(exc)
        checks.append('duplicate_photo_rejected')
        registry = app.registry()
        try:
            paths = [Path(row[0]) for row in registry.connection.execute('SELECT source_path FROM templates')]
        finally:
            registry.close()
        assert all(path.exists() for path in paths)
        checks.append('migration_source_photos_preserved')
        many = next(sorted(p.glob('*.png')) for p in (ROOT / 'data/muzzle_crops').iterdir() if len(list(p.glob('*.png'))) >= 4)
        uploads = [app.run('upload', item(path)) for path in many]
        result = app.run('enroll', {'kind': 'muzzle', 'images': uploads, 'cow_id': 'TEST-MANY', 'confirm_review': True})
        assert result['status'] == 'enrolled' and result['photo_count'] == len(many)
        assert len(result['previews']) == len(many)
        record = next(c for c in app.run('state')['cows'] if c['cow_id'] == 'TEST-MANY')
        assert len(record['photos']) == len(many)
        assert all(app.photo('TEST-MANY', i).startswith(b'\xff\xd8') for i in range(len(many)))
        checks.append('four_plus_photos_upload_enroll_and_gallery')
        assert all(not (ROOT / 'work/ui_uploads' / (upload['token'] + '.png')).exists() for upload in uploads)
        checks.append('consumed_uploads_removed')
        result = app.run('enroll', {'kind': 'muzzle', 'images': [item(crops[2])], 'cow_id': 'TEST-SINGLE'})
        assert result['status'] == 'enrollment_review_required' and result['single_photo']
        result = app.run('enroll', {'kind': 'muzzle', 'images': [item(crops[2])], 'cow_id': 'TEST-SINGLE', 'confirm_review': True})
        assert result['status'] == 'enrolled' and result['photo_count'] == 1
        checks.append('single_photo_review_and_enrollment')
        request = {'kind': 'muzzle', 'images': [item(crops[0])]}
        encoded = (np.zeros(app.system.dimension, dtype=np.float32), 'test', None, {'flags': []})
        candidate = {'cow_id': 'TEST-COW-8', 'similarity': 0.192, 'details': {'owner': 'Private owner'}}
        with patch.object(app.pipeline, 'encode_path', return_value=encoded), patch.object(CowRegistry, 'candidates', return_value=[candidate]):
            rejected = app.run('identify', request)
            assert rejected['status'] == 'no_confident_match' and rejected['candidates'] == []
            assert 'Private owner' not in json.dumps(rejected) and 'TEST-COW-8' not in json.dumps(rejected)
        checks.append('below_threshold_hides_cow_id_and_details')
        candidate['similarity'] = 0.99
        with patch.object(app.pipeline, 'encode_path', return_value=encoded), patch.object(CowRegistry, 'candidates', return_value=[candidate]):
            accepted = app.run('identify', request)
            assert accepted['status'] == 'candidate_match_requires_review'
            assert accepted['candidates'][0]['cow_id'] == 'TEST-COW-8'
        checks.append('accepted_match_returns_enrolled_details')
        with patch.object(app.pipeline, 'encode_path', return_value=(*encoded[:3], {'flags': ['blur']})), patch.object(CowRegistry, 'candidates', return_value=[candidate]):
            rejected = app.run('identify', request)
            assert rejected['status'] == 'capture_review_required' and rejected['candidates'] == []
        checks.append('poor_capture_hides_details_even_above_threshold')
    print(json.dumps({'passed': len(checks), 'checks': checks}, indent=2))

if __name__ == '__main__':
    main()
