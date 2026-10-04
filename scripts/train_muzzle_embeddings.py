"""Train ArcFace embeddings, select on validation cows, test once at the end."""
import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('TORCH_HOME',str(ROOT/'models/torch-cache'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/matplotlib'))
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from src.biometrics import MuzzleEncoder,ArcFace,SquarePad,image_transform,retrieval_protocol,rejection_threshold,summarize_protocol

class CropDataset(Dataset):
    def __init__(self,rows,classes=None,training=False):
        self.rows=rows;self.classes=classes;self.transform=image_transform(training)
        prepare=transforms.Compose([SquarePad(),transforms.Resize((224,224))])
        self.images=[]
        for row in rows:
            with Image.open(ROOT/'data/muzzle_crops'/row['image_path']) as im:self.images.append(prepare(im.convert('RGB')))
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        label=self.classes[self.rows[index]['animal_id']] if self.classes else index
        return self.transform(self.images[index]),label

def encode(model,loader,device):
    model.eval();vectors=[]
    with torch.inference_mode():
        for images,_ in loader:vectors.append(model(images.to(device)).cpu().numpy())
    return np.concatenate(vectors)

def save_checkpoint(path,payload):
    temporary=path.with_suffix('.tmp');torch.save(payload,temporary);os.replace(temporary,path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=40);parser.add_argument('--batch',type=int,default=16);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if not torch.cuda.is_available():raise SystemExit('CUDA required for this training command.')
    device='cuda';out=ROOT/'outputs/embeddings/resnet18_arcface'
    if out.exists() and not args.resume:raise SystemExit('Run exists; use --resume.')
    if args.resume and not (out/'last.pt').exists():raise SystemExit('No resume checkpoint.')
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'data/embedding/manifest.json').read_text());classes={c:i for i,c in enumerate(manifest['train_classes'])}
    train_rows=[r for r in manifest['rows'] if r['split']=='train'];val_rows=[r for r in manifest['rows'] if r['split']=='val']
    train=DataLoader(CropDataset(train_rows,classes,True),batch_size=args.batch,shuffle=True,num_workers=0)
    val=DataLoader(CropDataset(val_rows),batch_size=args.batch,num_workers=0)
    pretrained=ROOT/'models/resnet18-f37072fd.pth'
    if not pretrained.exists():
        state=torch.hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth',model_dir=str(ROOT/'models'),check_hash=True,map_location='cpu')
    model=MuzzleEncoder(pretrained).to(device);head=ArcFace(len(classes)).to(device)
    for name,param in model.backbone.named_parameters():param.requires_grad=name.startswith(('layer3','layer4'))
    optimizer=torch.optim.AdamW([{'params':model.projection.parameters(),'lr':0.001},
                                {'params':head.parameters(),'lr':0.001},
                                {'params':[p for p in model.backbone.parameters() if p.requires_grad],'lr':0.00003}],weight_decay=0.0001)
    scaler=torch.amp.GradScaler('cuda');loss_fn=nn.CrossEntropyLoss();best_score=-1;best_epoch=0;start=1;history=[]
    if args.resume:
        cp=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(cp['encoder']);head.load_state_dict(cp['head']);optimizer.load_state_dict(cp['optimizer']);scaler.load_state_dict(cp['scaler'])
        best_score=cp['best_score'];best_epoch=cp['best_epoch'];start=cp['epoch']+1;history=cp['history']
        torch.set_rng_state(cp['torch_rng']);torch.cuda.set_rng_state_all(cp['cuda_rng']);random.setstate(cp['python_rng'])
        ns=cp['numpy_rng'];np.random.set_state((ns[0],np.array(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
    baseline=retrieval_protocol(encode(model,val,device),val_rows)
    if not args.resume:
        (out/'initial_validation.json').write_text(json.dumps(summarize_protocol(baseline),indent=2))
        print('Initial validation rank-1:',baseline['closed_rank1'],flush=True)
    begin=time.monotonic()
    for epoch in range(start,args.epochs+1):
        model.train();head.train()
        # Keep ImageNet BN statistics stable with this small capture dataset.
        for module in model.backbone.modules():
            if isinstance(module,nn.BatchNorm2d):module.eval()
        lr_factor=float(0.5*(1+np.cos(np.pi*(epoch-1)/args.epochs)))
        for i,g in enumerate(optimizer.param_groups):g['lr']=([0.001,0.001,0 if epoch<=3 else 0.00003][i])*lr_factor
        margin=0.2*min(1,max(0,(epoch-3)/5));total_loss=0;count=0
        for images,labels in train:
            images=images.to(device);labels=labels.to(device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda'):
                embeddings=model(images)
            # Angular-margin arithmetic stays float32 for numerical stability.
            loss=loss_fn(head(embeddings.float(),labels,margin),labels)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),5)
            scaler.step(optimizer);scaler.update();total_loss+=loss.item()*len(labels);count+=len(labels)
        protocol=retrieval_protocol(encode(model,val,device),val_rows)
        # Rank-1 on unseen validation cows determines the checkpoint.
        score=protocol['closed_rank1'];improved=score>best_score
        if improved:best_score=score;best_epoch=epoch
        row={'epoch':epoch,'loss':total_loss/count,'validation_rank1':score,'best_epoch':best_epoch,'seconds':time.monotonic()-begin};history.append(row)
        metadata={'architecture':'resnet18_arcface_256','dimension':256,'input_size':224,'normalization':'ImageNet RGB, aspect-preserving square pad',
                  'train_classes':manifest['train_classes'],'seed':42,'epoch':epoch,'best_score':best_score,'best_epoch':best_epoch}
        if improved:save_checkpoint(out/'best.pt',{'encoder':model.state_dict(),**metadata})
        ns=np.random.get_state()
        save_checkpoint(out/'last.pt',{'encoder':model.state_dict(),'head':head.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                                    'history':history,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),
                                    'numpy_rng':(ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]),**metadata})
        with (out/'history.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
        print(json.dumps(row),flush=True)
        if epoch>=12 and epoch-best_epoch>=10:
            print('Early stopping: 10 epochs without unseen-cow rank-1 improvement.',flush=True);break
    cp=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(cp['encoder'])
    val_vectors=encode(model,val,device);v=retrieval_protocol(val_vectors,val_rows)
    pair_t=rejection_threshold(v['negative_pairs']);ident_t=rejection_threshold([r['score'] for r in v['unknown_queries']])
    calibration={'model_sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),'pair_threshold':pair_t,'identification_threshold':ident_t,'target_empirical_false_acceptance':0.01,
                 'validation':summarize_protocol(v,pair_t,ident_t),'warning':'Small dependent trials; no population FAR guarantee. 1:N threshold calibrated only for a 13-cow gallery, not statewide scale.'}
    (out/'calibration.json').write_text(json.dumps(calibration,indent=2))
    # Test rows are only loaded after checkpoint and thresholds are frozen.
    test_rows=[r for r in manifest['rows'] if r['split']=='test'];test=DataLoader(CropDataset(test_rows),batch_size=args.batch,num_workers=0)
    test_vectors=encode(model,test,device);t=retrieval_protocol(test_vectors,test_rows)
    report={'best_epoch':cp['epoch'],'epochs_completed':len(history),'validation':calibration['validation'],
            'test':summarize_protocol(t,pair_t,ident_t),'split_images':manifest['split_images'],'split_cows':manifest['split_cows'],
            'limitations':['Only 31 test cows; gallery folds repeat photos and are dependent.','Capture sessions unknown; same-session results may be optimistic.','Reviewed ground-truth crops used: detector misses and crop errors are not included in these identity metrics.','No statewide or 95% accuracy claim. Thresholds require field calibration and change with gallery scale.']}
    (out/'evaluation.json').write_text(json.dumps(report,indent=2))
    (out/'test_queries.json').write_text(json.dumps(t['closed_queries'],indent=2))
    np.savez(out/'test_embeddings.npz',embeddings=test_vectors,image_paths=np.array([r['image_path'] for r in test_rows]))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
