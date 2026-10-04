"""Train a small muzzle detector locally; reserve test cows for final evaluation."""
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
(ROOT/'work/ultralytics/Ultralytics').mkdir(parents=True, exist_ok=True)
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT/'work/ultralytics'))
os.environ.setdefault('WANDB_MODE','disabled')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'work/matplotlib'))

def main():
    import torch
    from ultralytics import YOLO, settings
    settings.update({k:False for k in ('sync','wandb','mlflow','comet','clearml','neptune') if k in settings})
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise SystemExit('CUDA unavailable; refusing unexpectedly slow CPU training.')
    run=ROOT/'outputs/yolo/muzzle_yolo11n'
    if args.resume:
        YOLO(str(run/'weights/last.pt')).train(resume=True)
    else:
        if run.exists(): raise SystemExit('Training run already exists. Use --resume to continue.')
        model=YOLO(str(ROOT/'models/yolo11n.pt'))
        model.train(data=str(ROOT/'data/yolo_muzzle/dataset.yaml'), epochs=args.epochs,
                    imgsz=640, batch=args.batch, device=0, workers=0, cache=False,
                    project=str(ROOT/'outputs/yolo'), name='muzzle_yolo11n',
                    seed=42, deterministic=True, patience=15, optimizer='AdamW', lr0=0.001,
                    amp=True, plots=True, save=True, save_period=10,
                    degrees=10, translate=0.1, scale=0.25, fliplr=0.5, flipud=0,
                    hsv_h=0.015, hsv_s=0.3, hsv_v=0.3, mosaic=0.5, close_mosaic=10,
                    mixup=0, cutmix=0, auto_augment=None)
    best=YOLO(str(run/'weights/best.pt'))
    metrics=best.val(data=str(ROOT/'data/yolo_muzzle/dataset.yaml'), split='test',
                     imgsz=640, batch=args.batch, device=0, workers=0, plots=True,
                     project=str(ROOT/'outputs/yolo'), name='held_out_test')
    report={'model':str(run/'weights/best.pt'), 'test_metrics':metrics.results_dict,
            'note':'Detection agreement with assistant-reviewed labels on held-out cows; not cow identification accuracy or production certification.'}
    (ROOT/'outputs/yolo/test_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
