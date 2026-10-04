"""Evaluate a fixed detection threshold and preview held-out muzzle crops."""
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'work/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/matplotlib'))

def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0

def main():
    from ultralytics import YOLO
    manifest=json.loads((ROOT/'data/yolo_muzzle/manifest.json').read_text())
    rows=[r for r in manifest['records'] if r['split']=='test']
    lookup={str((ROOT/'data/yolo_muzzle'/r['image']).resolve()):r for r in rows}
    preview_rows=rows[::max(1,len(rows)//12)][:12]
    preview_indices={str((ROOT/'data/yolo_muzzle'/r['image']).resolve()):i for i,r in enumerate(preview_rows)}
    canvas=Image.new('RGB',(960,660),'white');draw=ImageDraw.Draw(canvas)
    tp=fp=fn=0;records=[]
    model=YOLO(str(ROOT/'outputs/yolo/muzzle_yolo11n/weights/best.pt'))
    for result in model.predict(source=list(lookup),stream=True,batch=4,device=0,imgsz=640,conf=0.25,verbose=False):
        key=str(Path(result.path).resolve());row=lookup[key]
        boxes=result.boxes.xyxy.cpu().tolist();scores=result.boxes.conf.cpu().tolist()
        overlaps=[iou(box,row['box']) for box in boxes]
        matched=any(v>=0.5 for v in overlaps)
        tp+=int(matched);fn+=int(not matched);fp+=len(boxes)-int(matched)
        records.append({'source_path':row['source_path'],'animal_id':row['animal_id'],
                        'predicted_boxes':boxes,'confidences':scores,'label_ious':overlaps})
        if key in preview_indices:
            j=preview_indices[key];x=j%4*240;y=j//4*220
            if len(boxes)==1:
                with Image.open(result.path) as im:
                    crop=im.crop(tuple(round(v) for v in boxes[0])).convert('RGB');crop.thumbnail((230,180));canvas.paste(crop,(x,y))
                text=f"{row['animal_id']} conf={scores[0]:.2f}"
            else:text=f"{row['animal_id']}: {len(boxes)} detections"
            draw.text((x,y+184),text,fill='black')
            draw.text((x,y+199),'held-out test photo',fill='black')
    out=ROOT/'outputs/yolo';canvas.save(out/'test_crop_preview.jpg')
    report={'confidence_threshold':0.25,'iou_threshold':0.5,'test_images':len(rows),
            'true_positive_boxes':tp,'false_positive_boxes':fp,'missed_labels':fn,
            'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn) if tp+fn else 0,
            'zero_detection_images':sum(not r['predicted_boxes'] for r in records),
            'multiple_detection_images':sum(len(r['predicted_boxes'])>1 for r in records),
            'note':'Threshold fixed at 0.25 before test inference; not field-calibrated. Every test image has a labeled muzzle, so this cannot estimate false alarms on no-muzzle images. Label agreement, not identity accuracy.',
            'images':records}
    (out/'fixed_threshold_test.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='images'},indent=2))

if __name__=='__main__':main()
