"""Controlled MobileNet/CosFace experiments; active model and registry untouched."""
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
from src.biometrics import SquarePad,normalize,rejection_threshold,summarize_protocol
from src.hybrid_biometrics import split_normalize
from src.identity_system import IdentitySystem

OUT=ROOT/'outputs/mobilenet_cosface'
VIEWS=('full','left_half','right_half')

def view_crop(image,view):
    w,h=image.size
    if view=='left_half':return image.crop((0,0,max(1,w//2),h))
    if view=='right_half':return image.crop((w//2,0,w,h))
    if view!='full':raise ValueError(view)
    return image

class Photos(Dataset):
    def __init__(self,rows,size,classes=None,training=False,partial=False,view='full'):
        self.rows=rows;self.classes=classes;self.training=training;self.partial=partial;self.view=view
        self.images=[]
        for r in rows:
            with Image.open(ROOT/'data/muzzle_crops'/r['image_path']) as im:self.images.append(im.convert('RGB').copy())
        ops=[SquarePad(),transforms.Resize((size,size))]
        if training:ops += [transforms.RandomAffine(10,translate=(.04,.04),scale=(.95,1.05),fill=(123,116,103)),transforms.ColorJitter(.15,.15,.1)]
        self.transform=transforms.Compose(ops+[transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        image=self.images[i];view=self.view
        if self.training and self.partial and random.random()<.35:view=random.choice(VIEWS[1:])
        return self.transform(view_crop(image,view)),self.classes[self.rows[i]['animal_id']] if self.classes else i

class MobileEncoder(nn.Module):
    def __init__(self,pretrained=False):
        super().__init__();self.backbone=models.mobilenet_v3_large(weights=None)
        if pretrained:self.backbone.load_state_dict(torch.load(ROOT/'models/mobilenet_v3_large-5c1a4163.pth',weights_only=True,map_location='cpu'))
        self.backbone.classifier=nn.Identity();self.projection=nn.Linear(960,256,bias=False);nn.init.orthogonal_(self.projection.weight)
    def forward(self,x):return F.normalize(self.projection(self.backbone(x)),dim=1)

class CosFace(nn.Module):
    def __init__(self,classes):
        super().__init__();self.weight=nn.Parameter(torch.empty(classes,256));nn.init.xavier_uniform_(self.weight)
    def forward(self,x,labels,margin=.35):
        cosine=F.linear(F.normalize(x,dim=1),F.normalize(self.weight,dim=1))
        return 30*(cosine-margin*F.one_hot(labels,len(self.weight)))

@torch.inference_mode()
def encode(model,rows,size,view='full'):
    model.eval();parts=[]
    for images,_ in DataLoader(Photos(rows,size,view=view),batch_size=8):parts.append(model(images.cuda()).float().cpu().numpy())
    return np.concatenate(parts)

def protocol(gallery,query,rows,dimensions,weights):
    """Full-image enrollment; independent full/partial query photos, fixed 3 folds."""
    cows=sorted({r['animal_id'] for r in rows},key=lambda c:int(c.split('_')[1]));indices={c:[i for i,r in enumerate(rows) if r['animal_id']==c] for c in cows}
    shuffled=cows.copy();np.random.default_rng(42).shuffle(shuffled);known=set(shuffled[:len(cows)//2]);kp=[i for i,c in enumerate(cows) if c in known]
    unknown=set(cows)-known;closed=[];kq=[];uq=[]
    gs=split_normalize(gallery,dimensions);qs=split_normalize(query,dimensions)
    for fold in range(3):
        chosen=[[idx[fold%len(idx)],idx[(fold+1)%len(idx)]] for idx in indices.values()]
        scores=np.zeros((len(rows),len(cows)))
        for g,q,weight in zip(gs,qs,weights):
            templates=g[np.array(chosen)];centers=normalize(templates.mean(axis=1));maximum=np.einsum('nd,ckd->nck',q,templates).max(axis=2)
            scores+=weight*.5*(q@centers.T+maximum)
        for ci,c in enumerate(cows):
            for i in indices[c]:
                if i in chosen[ci]:continue
                winner=int(scores[i].argmax());closed.append({'animal_id':c,'correct':cows[winner]==c,'score':float(scores[i,winner])})
                if c in known:
                    winner=kp[int(scores[i,kp].argmax())];kq.append({'animal_id':c,'correct':cows[winner]==c,'score':float(scores[i,winner])})
        for c in sorted(unknown):
            for i in indices[c]:uq.append({'animal_id':c,'score':float(scores[i,kp].max())})
    # One deterministic orientation per unordered pair; never compare a photo with itself.
    matrix=sum(w*(g@q.T) for g,q,w in zip(gs,qs,weights));pos=[];neg=[]
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):(pos if rows[i]['animal_id']==rows[j]['animal_id'] else neg).append(float(matrix[i,j]))
    return {'cow_count':len(cows),'image_count':len(rows),'closed_rank1':float(np.mean([q['correct'] for q in closed])), 'closed_queries':closed,'known_gallery_cows':len(known),'unknown_cows':len(unknown),'known_queries':kq,'unknown_queries':uq,'positive_pairs':pos,'negative_pairs':neg}

def calibrate(p):
    pt=rejection_threshold(p['negative_pairs']);it=rejection_threshold([q['score'] for q in p['unknown_queries']]);wrong=[q['score'] for q in p['known_queries'] if not q['correct']]
    if wrong:it=max(it,float(np.nextafter(max(wrong),np.inf)))
    # Balanced pair accuracy operating point, separate from security threshold.
    pos=np.array(p['positive_pairs']);neg=np.array(p['negative_pairs']);values=np.unique(np.concatenate([pos,neg]));thresholds=np.concatenate([values,[np.nextafter(values[-1],np.inf)]])
    bt=max(thresholds,key=lambda t:.5*(np.mean(pos>=t)+np.mean(neg<t)))
    return {'pair_threshold':pt,'identification_threshold':it,'balanced_pair_threshold':float(bt)}

def metrics(p,cal):
    result=summarize_protocol(p,cal['pair_threshold'],cal['identification_threshold']);pos=np.array(p['positive_pairs']);neg=np.array(p['negative_pairs']);t=cal['balanced_pair_threshold']
    result['pair_verification']['balanced_accuracy']=float(.5*(np.mean(pos>=t)+np.mean(neg<t)))
    result['pair_verification']['balanced_accuracy_threshold']=t
    return result

def selection_score(m):return .5*(m['closed_rank1']+m['open_identification']['known_correct_acceptance'])

def baseline(rows):
    directory=ROOT/json.loads((ROOT/'outputs/embeddings/active_model.json').read_text())['model_dir'];system=IdentitySystem(directory,'cuda')
    vectors={}
    for view in VIEWS:vectors[view]=np.concatenate([encode(model,rows,224,view) for model in system.models],axis=1)
    spec={'dimensions':system.dimensions,'weights':system.weights,'model_hash':system.model_hash};del system;torch.cuda.empty_cache()
    return vectors,spec

def train(config,train_rows,val_rows,classes,epochs):
    run=OUT/config['name'];run.mkdir(parents=True,exist_ok=True)
    if (run/'result.json').exists():return json.loads((run/'result.json').read_text())
    if (run/'history.json').exists():raise ValueError('Incomplete run preserved; use a new output directory rather than overwrite')
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model=MobileEncoder(True).cuda();head=CosFace(len(classes)).cuda()
    # Stable small-data fine-tuning: final feature blocks, frozen BatchNorm statistics.
    for name,p in model.backbone.named_parameters():p.requires_grad=name.startswith(('features.12.','features.13.','features.14.','features.15.','features.16.'))
    optimizer=torch.optim.AdamW([{'params':[p for p in model.backbone.parameters() if p.requires_grad],'lr':3e-5},{'params':model.projection.parameters(),'lr':3e-4},{'params':head.parameters(),'lr':5e-4}],weight_decay=.001)
    scaler=torch.amp.GradScaler('cuda');dataset=Photos(train_rows,config['size'],classes,True,config['partial']);grouped={c:[i for i,r in enumerate(train_rows) if r['animal_id']==c] for c in classes}
    best=-1;best_epoch=0;history=[];begin=time.monotonic()
    for epoch in range(1,epochs+1):
        model.train();head.train()
        for m in model.backbone.modules():
            if isinstance(m,nn.BatchNorm2d):m.eval()
        factor=.5*(1+math.cos(math.pi*(epoch-1)/epochs))
        for g,lr in zip(optimizer.param_groups,[0 if epoch<=2 else 3e-5,3e-4,5e-4]):g['lr']=lr*factor
        cows=list(classes);random.shuffle(cows);losses=[]
        for start in range(0,len(cows),4):
            subset=cows[start:start+4]
            if len(subset)<2:continue
            batch=[dataset[i] for c in subset for i in random.sample(grouped[c],3)];images=torch.stack([b[0] for b in batch]).cuda();labels=torch.tensor([b[1] for b in batch],device='cuda')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):embedding=model(images)
            margin=.35*min(1,epoch/5);loss=F.cross_entropy(head(embedding.float(),labels,margin),labels)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),5);scaler.step(optimizer);scaler.update();losses.append(loss.item())
        vectors=encode(model,val_rows,config['size']);p=protocol(vectors,vectors,val_rows,[256],[1.]);cal=calibrate(p);m=metrics(p,cal);score=selection_score(m)
        if score>best:
            best=score;best_epoch=epoch
            torch.save({'encoder':model.state_dict(),'architecture':'mobilenet_v3_large_cosface','dimension':256,'input_size':config['size'],'config':config,'epoch':epoch,'seed':42},run/'best.pt')
        row={'epoch':epoch,'loss':float(np.mean(losses)),'rank1':m['closed_rank1'],'known_correct_acceptance':m['open_identification']['known_correct_acceptance'],'score':score,'best_epoch':best_epoch,'seconds':time.monotonic()-begin};history.append(row);(run/'history.json').write_text(json.dumps(history,indent=2));print(config['name'],json.dumps(row),flush=True)
        if epoch>=15 and epoch-best_epoch>=10:break
    model.load_state_dict(torch.load(run/'best.pt',weights_only=True,map_location='cpu')['encoder']);views={v:encode(model,val_rows,config['size'],v) for v in VIEWS};p=protocol(views['full'],views['full'],val_rows,[256],[1.]);cal=calibrate(p)
    report={'config':config,'best_epoch':best_epoch,'epochs_completed':len(history),'calibration':cal,'validation':{v:metrics(protocol(views['full'],views[v],val_rows,[256],[1.]),cal) for v in VIEWS}}
    np.savez(run/'validation_embeddings.npz',**views);(run/'result.json').write_text(json.dumps(report,indent=2));del model,head,optimizer;torch.cuda.empty_cache();return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=35);args=parser.parse_args();torch.set_num_threads(4)
    if not torch.cuda.is_available():raise SystemExit('CUDA required')
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'comparison.json').exists():raise SystemExit('Completed experiment exists; no overwrite.')
    manifest=json.loads((ROOT/'data/embedding/manifest.json').read_text());tr=[r for r in manifest['rows'] if r['split']=='train'];va=[r for r in manifest['rows'] if r['split']=='val'];classes={c:i for i,c in enumerate(manifest['train_classes'])}
    configs=[{'name':'mobile224','size':224,'partial':False},{'name':'mobile384','size':384,'partial':False},{'name':'mobile384_partial','size':384,'partial':True}]
    plan={'configs':configs,'epochs_max':args.epochs,'seed':42,'batch':12,'loss':'CosFace m=.35 s=30; margin warmup five epochs','optimizer':'AdamW; late feature-block fine-tuning; frozen BatchNorm','paper_replication':False,'matching':'hybrid; two enrollment photos; three fixed folds','partial_query':'synthetic left/right half; gallery is full and excludes the source query image','calibration':'validation only; fixed thresholds transferred unchanged to held-out test','test':'compare frozen baseline and validation-selected candidate once; do not tune on test','keypoint_alignment':'Not run: no verified nostril keypoint labels exist. Existing detector predicts boxes only.','deployment':'active bundle and registry untouched'}
    (OUT/'plan.json').write_text(json.dumps(plan,indent=2))
    if (OUT/'baseline_validation.json').exists():base=json.loads((OUT/'baseline_validation.json').read_text())
    else:
        vectors,spec=baseline(va);p=protocol(vectors['full'],vectors['full'],va,spec['dimensions'],spec['weights']);cal=calibrate(p);base={'spec':spec,'calibration':cal,'validation':{v:metrics(protocol(vectors['full'],vectors[v],va,spec['dimensions'],spec['weights']),cal) for v in VIEWS}}
        np.savez(OUT/'baseline_validation_embeddings.npz',**vectors);(OUT/'baseline_validation.json').write_text(json.dumps(base,indent=2));print('baseline_validation',json.dumps(base),flush=True)
    reports=[]
    for config in configs:reports.append(train(config,tr,va,classes,args.epochs))
    eligible=[r for r in reports if r['validation']['full']['closed_rank1']>=base['validation']['full']['closed_rank1'] and r['validation']['full']['open_identification']['known_correct_acceptance']>=base['validation']['full']['open_identification']['known_correct_acceptance']]
    winner=max(eligible or reports,key=lambda r:selection_score(r['validation']['full']));selection={'candidate':winner['config']['name'],'eligible_vs_baseline':bool(eligible),'decision_before_test':True}
    (OUT/'selection.json').write_text(json.dumps(selection,indent=2))
    # Test rows and image pixels are first decoded here, after selection is locked.
    te=[r for r in manifest['rows'] if r['split']=='test'];vectors,spec=baseline(te);test={'baseline':{v:metrics(protocol(vectors['full'],vectors[v],te,spec['dimensions'],spec['weights']),base['calibration']) for v in VIEWS}}
    model=MobileEncoder().cuda();model.load_state_dict(torch.load(OUT/winner['config']['name']/'best.pt',weights_only=True,map_location='cpu')['encoder']);vectors={v:encode(model,te,winner['config']['size'],v) for v in VIEWS};test['candidate']={v:metrics(protocol(vectors['full'],vectors[v],te,[256],[1.]),winner['calibration']) for v in VIEWS}
    result={'plan':plan,'baseline':base,'candidates':reports,'selection':selection,'held_out_test':test,'warning':'Small identity-disjoint test; dependent gallery folds; synthetic partial crops. Historical baseline test use means this is not a newly collected prospective test. No statewide error guarantee.'}
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2));print('COMPLETED',json.dumps({'selection':selection,'held_out_test':test}),flush=True)

if __name__=='__main__':main()
