"""Load a single encoder or an immutable CNN/ViT scoring bundle."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from src.biometrics import image_transform,load_encoder
from src.hybrid_biometrics import fused_pair_score

def bundle_hash(directory,descriptor):
    directory=Path(directory);content={'descriptor':descriptor,'checkpoint_hashes':[]}
    for branch in descriptor['branches']:
        path=(directory/branch['checkpoint']).resolve()
        if not path.is_relative_to(directory.resolve()):raise ValueError('Checkpoint path escapes model directory')
        content['checkpoint_hashes'].append(hashlib.sha256(path.read_bytes()).hexdigest())
    return hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class IdentitySystem:
    def __init__(self,directory,device='cpu',load_models=True):
        self.directory=Path(directory);self.device=device;self.transform=image_transform();self.models=[]
        self.calibration=json.loads((self.directory/'calibration.json').read_text())
        system=self.directory/'system.json'
        if system.exists():
            descriptor=json.loads(system.read_text())
            if descriptor['format_version']!=1:raise ValueError('Unsupported system version')
            if descriptor.get('preprocessing','rgb_squarepad224_imagenet_v1')!='rgb_squarepad224_imagenet_v1':raise ValueError('Unsupported preprocessing version')
            self.model_hash=bundle_hash(self.directory,descriptor)
            self.dimensions=[b['dimension'] for b in descriptor['branches']];self.weights=descriptor['weights']
            self.names=[b['name'] for b in descriptor['branches']]
            self.matching_mode=descriptor['matching_mode']
            if len(self.weights)!=len(self.dimensions) or not np.isclose(sum(self.weights),1) or min(self.weights)<0:raise ValueError('Invalid fusion weights')
            paths=[self.directory/b['checkpoint'] for b in descriptor['branches']]
        else:
            path=self.directory/'best.pt';self.model_hash=hashlib.sha256(path.read_bytes()).hexdigest()
            self.dimensions=[256];self.weights=[1.0];self.names=['cnn'];self.matching_mode=self.calibration.get('matching_mode','centroid');paths=[path]
        if self.calibration['model_sha256']!=self.model_hash:raise ValueError('Calibration does not match model bundle')
        if self.calibration.get('matching_mode',self.matching_mode)!=self.matching_mode:raise ValueError('Calibration scoring mismatch')
        self.dimension=sum(self.dimensions)
        if load_models:
            for path,dimension in zip(paths,self.dimensions):
                model,checkpoint=load_encoder(path,device)
                if checkpoint.get('dimension',256)!=dimension:raise ValueError('Encoder dimension mismatch')
                self.models.append(model)
    def embed(self,image):
        if not self.models:raise ValueError('Encoders were not loaded')
        image=self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():parts=[m(image)[0].float().cpu().numpy() for m in self.models]
        return np.concatenate(parts).astype(np.float32)
    def pair_score(self,a,b):return fused_pair_score(a,b,self.dimensions,self.weights)
