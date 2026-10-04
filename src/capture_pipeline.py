"""Shared full-image or cropped-photo capture path for lookup and migration."""
import hashlib
import json
from PIL import Image,ImageOps

class CapturePipeline:
    def __init__(self,system,detector_weights):
        self.system=system;self.detector_weights=detector_weights;self.detector=None
        path=system.directory/'capture_policy.json'
        self.policy=json.loads(path.read_text()) if path.exists() else None
    def encode_path(self,path,cropped=False):
        with Image.open(path) as source:image=ImageOps.exif_transpose(source).convert('RGB')
        digest=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest();box=None
        if not cropped:
            if self.detector is None:
                from ultralytics import YOLO
                self.detector=YOLO(str(self.detector_weights))
            result=self.detector.predict(image,imgsz=640,conf=0.25,device=self.system.device,verbose=False)[0]
            if len(result.boxes)!=1:raise ValueError(f'{path.name}: {len(result.boxes)} muzzles detected; recapture or review.')
            box=[round(v) for v in result.boxes.xyxy[0].cpu().tolist()]
            if min(box[2]-box[0],box[3]-box[1])<32:raise ValueError('Muzzle too small; take a closer photo.')
            image=image.crop(tuple(box))
        quality={'flags':[]}
        if self.policy:
            from src.capture_quality import inspect_capture
            quality=inspect_capture(image,self.policy)
        return self.system.embed(image),digest,box,quality
