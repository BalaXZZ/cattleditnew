"""Check trained artifacts and run both embedding branches on a synthetic image."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'work/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'work/matplotlib'))

def main():
    from PIL import Image
    import numpy as np
    import torch
    from ultralytics import YOLO
    from src.identity_system import IdentitySystem
    torch.set_num_threads(4)
    pointer = ROOT / 'outputs/embeddings/active_model.json'
    if not pointer.exists():
        raise SystemExit('Install the team runtime bundle first; see README.md.')
    system = IdentitySystem(ROOT / json.loads(pointer.read_text())['model_dir'], 'cpu')
    embedding = system.embed(Image.new('RGB', (224, 224), (128, 128, 128)))
    assert embedding.shape == (512,) and np.isfinite(embedding).all()
    YOLO(str(ROOT / 'outputs/yolo/muzzle_yolo11n/weights/best.pt'))
    print(f'CPU runtime OK: {system.names}, {embedding.size} values, YOLO loaded.')
    print('Synthetic smoke check only; this does not measure identification accuracy.')

if __name__ == '__main__':
    main()
