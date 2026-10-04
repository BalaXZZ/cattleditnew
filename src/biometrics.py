"""Versioned, normalized muzzle embeddings and angular-margin training."""
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
from torchvision.transforms import functional as TF

class SquarePad:
    def __call__(self, image):
        w,h=image.size;size=max(w,h)
        return TF.pad(image,((size-w)//2,(size-h)//2,size-w-(size-w)//2,size-h-(size-h)//2),fill=(123,116,103))

def image_transform(training=False):
    steps=[SquarePad(),transforms.Resize((224,224))]
    if training:
        steps.extend([transforms.RandomAffine(10,translate=(0.04,0.04),scale=(0.95,1.05),fill=(123,116,103)),
                      transforms.ColorJitter(brightness=0.15,contrast=0.15,saturation=0.1)])
    steps.extend([transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    return transforms.Compose(steps)

class GeMPool(nn.Module):
    def forward(self,x):
        return x.float().clamp_min(1e-6).pow(3.0).mean(dim=(-2,-1),keepdim=True).pow(1/3)

class MuzzleEncoder(nn.Module):
    def __init__(self,pretrained_path=None,pooling='avg'):
        super().__init__()
        self.backbone=models.resnet18(weights=None)
        if pretrained_path:
            self.backbone.load_state_dict(torch.load(pretrained_path,map_location='cpu',weights_only=True))
        self.backbone.fc=nn.Identity()
        if pooling=='gem':self.backbone.avgpool=GeMPool()
        elif pooling!='avg':raise ValueError('Unknown pooling')
        self.projection=nn.Linear(512,256,bias=False)
        nn.init.orthogonal_(self.projection.weight)
    def forward(self,images):
        return F.normalize(self.projection(self.backbone(images)),dim=1)

class ArcFace(nn.Module):
    def __init__(self,classes,scale=30.0,margin=0.2):
        super().__init__();self.weight=nn.Parameter(torch.empty(classes,256));nn.init.xavier_uniform_(self.weight)
        self.scale=scale;self.margin=margin
    def forward(self,embedding,labels,margin=None):
        m=self.margin if margin is None else margin
        cosine=F.linear(embedding,F.normalize(self.weight,dim=1)).clamp(-1+1e-7,1-1e-7)
        sine=(1-cosine.square()).clamp_min(1e-7).sqrt()
        phi=cosine*math.cos(m)-sine*math.sin(m)
        phi=torch.where(cosine>math.cos(math.pi-m),phi,cosine-math.sin(math.pi-m)*m)
        target=F.one_hot(labels,num_classes=self.weight.shape[0]).bool()
        return torch.where(target,phi,cosine)*self.scale

def load_encoder(path,device='cpu'):
    checkpoint=torch.load(Path(path),map_location='cpu',weights_only=True)
    if checkpoint.get('architecture','').startswith('dinov2'):
        from src.hybrid_biometrics import DinoMuzzleEncoder
        model=DinoMuzzleEncoder(config=checkpoint['vit_config'],feature_mode=checkpoint['feature_mode'],projection_dim=checkpoint['projection_dim'])
    else:model=MuzzleEncoder(pooling=checkpoint.get('pooling','avg'))
    model.load_state_dict(checkpoint['encoder']);model.to(device).eval()
    return model,checkpoint

def normalize(array):
    return array/np.maximum(np.linalg.norm(array,axis=-1,keepdims=True),1e-12)

def retrieval_protocol(embeddings,rows,score_mode='centroid'):
    """Three predetermined two-photo enrollment folds; no augmented test images."""
    cows=sorted({r['animal_id'] for r in rows},key=lambda x:int(x.split('_')[1]))
    indices={c:[i for i,r in enumerate(rows) if r['animal_id']==c] for c in cows}
    shuffled=cows.copy();np.random.default_rng(42).shuffle(shuffled)
    known=set(shuffled[:len(cows)//2]);unknown=set(cows)-known
    closed=[];known_queries=[];unknown_queries=[]
    for fold in range(3):
        gallery=[];templates=[];queries=[]
        for c in cows:
            idx=indices[c];chosen=[idx[fold%len(idx)],idx[(fold+1)%len(idx)]]
            gallery.append(normalize(embeddings[chosen].mean(axis=0)))
            templates.append(embeddings[chosen])
            queries.extend(i for i in idx if i not in chosen)
        gallery=np.stack(gallery);templates=np.stack(templates);known_positions=[i for i,c in enumerate(cows) if c in known]
        def score_query(vector):
            if score_mode=='centroid':return gallery@vector
            maximum=(templates@vector).max(axis=1)
            if score_mode=='max_template':return maximum
            if score_mode=='hybrid':return 0.5*(maximum+gallery@vector)
            raise ValueError('Unknown gallery scoring mode')
        for i in queries:
            scores=score_query(embeddings[i]);rank=np.argsort(-scores)
            true=rows[i]['animal_id'];pred=cows[int(rank[0])]
            closed.append({'fold':fold,'source_path':rows[i]['image_path'],'animal_id':true,'predicted_id':pred,'similarity':float(scores[rank[0]]),'correct':pred==true})
            if true in known:
                subset=scores[known_positions];order=np.argsort(-subset);winner=cows[known_positions[int(order[0])]]
                known_queries.append({'score':float(subset[order[0]]),'correct':winner==true,'animal_id':true})
        for c in sorted(unknown):
            for i in indices[c]:
                score=float(np.max(score_query(embeddings[i])[known_positions]))
                unknown_queries.append({'score':score,'animal_id':c})
    # All actual different-photo pairs. Pairs are dependent, not independent trials.
    scores=embeddings@embeddings.T;positive=[];negative=[]
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            (positive if rows[i]['animal_id']==rows[j]['animal_id'] else negative).append(float(scores[i,j]))
    return {'cow_count':len(cows),'image_count':len(rows),'closed_rank1':float(np.mean([r['correct'] for r in closed])),
            'closed_queries':closed,'known_gallery_cows':len(known),'unknown_cows':len(unknown),
            'known_queries':known_queries,'unknown_queries':unknown_queries,'positive_pairs':positive,'negative_pairs':negative}

def rejection_threshold(scores,target_far=0.01):
    """Conservative empirical threshold; no population FAR guarantee."""
    ordered=sorted(scores,reverse=True);allowed=int(len(ordered)*target_far)
    return float(np.nextafter(ordered[min(allowed,len(ordered)-1)],np.inf))

def summarize_protocol(result,pair_threshold=None,identification_threshold=None):
    report={k:result[k] for k in ('cow_count','image_count','closed_rank1','known_gallery_cows','unknown_cows')}
    report['closed_query_trials']=len(result['closed_queries'])
    if pair_threshold is not None:
        report['pair_verification']={'threshold':pair_threshold,
            'same_cow_acceptance':float(np.mean(np.array(result['positive_pairs'])>=pair_threshold)),
            'different_cow_false_acceptance':float(np.mean(np.array(result['negative_pairs'])>=pair_threshold)),
            'positive_pairs':len(result['positive_pairs']),'negative_pairs':len(result['negative_pairs'])}
    if identification_threshold is not None:
        k=result['known_queries'];u=result['unknown_queries']
        report['open_identification']={'threshold':identification_threshold,
            'known_correct_acceptance':float(np.mean([r['correct'] and r['score']>=identification_threshold for r in k])),
            'known_false_match_rate':float(np.mean([not r['correct'] and r['score']>=identification_threshold for r in k])),
            'unknown_false_acceptance':float(np.mean([r['score']>=identification_threshold for r in u])),
            'known_query_trials':len(k),'unknown_query_trials':len(u)}
    return report
