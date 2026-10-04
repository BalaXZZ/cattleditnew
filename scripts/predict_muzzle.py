"""Run the trained muzzle detector on one image and save an overlay and crop."""
import argparse
import json
import os
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'work/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/matplotlib'))

def main():
    from ultralytics import YOLO
    parser=argparse.ArgumentParser()
    parser.add_argument('image',type=Path)
    parser.add_argument('--weights',type=Path,default=ROOT/'outputs/yolo/muzzle_yolo11n/weights/best.pt')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/muzzle_prediction')
    parser.add_argument('--confidence',type=float,default=0.25)
    parser.add_argument('--device',default=None,help='cpu or GPU index, e.g. 0')
    args=parser.parse_args()
    if args.output.exists(): raise SystemExit('Output exists; choose a new --output directory.')
    with Image.open(args.image) as source: image=ImageOps.exif_transpose(source).convert('RGB')
    result=YOLO(str(args.weights)).predict(image,imgsz=640,conf=args.confidence,device=args.device,verbose=False)[0]
    args.output.mkdir(parents=True)
    boxes=result.boxes;detections=[];overlay=image.copy();draw=ImageDraw.Draw(overlay)
    for box,score in zip(boxes.xyxy.cpu().tolist(),boxes.conf.cpu().tolist()):
        coords=[round(v) for v in box]
        draw.rectangle(coords,outline='lime',width=3)
        draw.text((coords[0],max(0,coords[1]-15)),f'muzzle {score:.3f}',fill='lime')
        detections.append({'box':coords,'confidence':score})
    overlay.save(args.output/'detections.jpg',quality=95)
    # Never silently choose a cow when multiple muzzles are detected.
    if len(detections)==1:
        image.crop(tuple(detections[0]['box'])).save(args.output/'muzzle.png')
    report={'source':str(args.image.resolve()),'weights':str(args.weights.resolve()),
            'detections':detections,'status':'crop_saved' if len(detections)==1 else 'no_muzzle' if not detections else 'multiple_muzzles_review_required',
            'note':'Detection confidence is not cow-match confidence. Review crop visibility before biometric use.'}
    (args.output/'prediction.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
