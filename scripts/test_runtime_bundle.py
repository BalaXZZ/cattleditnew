"""Validate trusted-model installation without private data or real weights."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import zipfile
from runtime_bundle import FILES, digest, export_bundle, install_bundle

class BundleTests(unittest.TestCase):
    def make_bundle(self, root):
        for name in FILES:
            path = root / 'source' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'{"model_dir":"outputs/embeddings/cnn_vit_v3"}'
                             if name == FILES[0] else b'test-weight')
        bundle = root / 'bundle.zip'
        export_bundle(bundle, root / 'source')
        return bundle, digest(bundle.read_bytes())

    def test_roundtrip_preserves_registry_and_refuses_replacement(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            bundle, checksum = self.make_bundle(root)
            dest = root / 'clone'
            registry = dest / 'data/registry/cows.sqlite3'
            registry.parent.mkdir(parents=True)
            registry.write_bytes(b'private-registry')
            install_bundle(bundle, checksum, dest)
            self.assertEqual(registry.read_bytes(), b'private-registry')
            self.assertEqual((dest / FILES[-1]).read_bytes(), b'test-weight')
            (dest / FILES[-1]).write_bytes(b'other-model')
            with self.assertRaises(ValueError):
                install_bundle(bundle, checksum, dest)
            self.assertEqual((dest / FILES[-1]).read_bytes(), b'other-model')

    def test_checksum_and_unexpected_paths_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            bundle, checksum = self.make_bundle(root)
            dest = root / 'clone'
            with self.assertRaises(ValueError):
                install_bundle(bundle, '0' * 64, dest)
            with zipfile.ZipFile(bundle, 'a') as archive:
                archive.writestr('../escaped.py', b'bad')
            with self.assertRaises(ValueError):
                install_bundle(bundle, digest(bundle.read_bytes()), dest)
            self.assertFalse(dest.exists())

if __name__ == '__main__':
    unittest.main()
