"""DINOv2 muzzle features and separate-branch score fusion."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from src.biometrics import normalize

class DinoMuzzleEncoder(nn.Module):
    def __init__(self,config=None,pretrained_path=None,feature_mode='cls_patch',projection_dim=256):
        super().__init__()
        from transformers import Dinov2Config,Dinov2Model
        self.backbone=Dinov2Model.from_pretrained(pretrained_path,local_files_only=True,use_safetensors=True) if pretrained_path else Dinov2Model(Dinov2Config.from_dict(config))
        if feature_mode not in ('cls','cls_patch'):raise ValueError('Invalid ViT feature mode')
        self.feature_mode=feature_mode
        width=self.backbone.config.hidden_size*(2 if feature_mode=='cls_patch' else 1)
        self.projection=nn.Linear(width,projection_dim,bias=False) if projection_dim else nn.Identity()
        if projection_dim:nn.init.orthogonal_(self.projection.weight)
        self.dimension=projection_dim or width
    def forward(self,images):
        states=self.backbone(pixel_values=images).last_hidden_state
        features=states[:,0] if self.feature_mode=='cls' else torch.cat((states[:,0],states[:,1:].mean(dim=1)),dim=1)
        return F.normalize(self.projection(features),dim=1)

def split_normalize(vectors,dimensions):
    parts=np.split(vectors,np.cumsum(dimensions)[:-1],axis=-1)
    return [normalize(part) for part in parts]

def template_score(templates,query,mode='hybrid'):
    center=normalize(np.mean(templates,axis=0));centroid=float(center@query)
    maximum=float(np.max(templates@query))
    return centroid if mode=='centroid' else maximum if mode=='max_template' else 0.5*(centroid+maximum)

def fused_template_score(templates,query,dimensions,weights,mode='hybrid'):
    if len(dimensions)!=len(weights) or not np.isclose(sum(weights),1) or min(weights)<0:raise ValueError('Invalid branch weights')
    if sum(dimensions)!=len(query):raise ValueError('Branch dimensions do not match vector')
    template_parts=split_normalize(np.asarray(templates),dimensions);query_parts=split_normalize(np.asarray(query),dimensions)
    return float(sum(w*template_score(t,q,mode) for w,t,q in zip(weights,template_parts,query_parts)))

def fused_pair_score(a,b,dimensions,weights):
    return float(sum(w*float(x@y) for w,x,y in zip(weights,split_normalize(a,dimensions),split_normalize(b,dimensions))))

def fusion_protocol(branches,rows,weights,mode='hybrid'):
    """Same fixed two-photo gallery protocol, with independent branch scoring."""
    cows=sorted({r['animal_id'] for r in rows},key=lambda c:int(c.split('_')[1]));indices={c:[i for i,r in enumerate(rows) if r['animal_id']==c] for c in cows}
    shuffled=cows.copy();np.random.default_rng(42).shuffle(shuffled);known=set(shuffled[:len(cows)//2]);unknown=set(cows)-known
    closed=[];kq=[];uq=[];dimensions=[b.shape[1] for b in branches];vectors=np.concatenate(branches,axis=1)
    for fold in range(3):
        galleries={};queries=[]
        for c,idx in indices.items():
            chosen=[idx[fold%len(idx)],idx[(fold+1)%len(idx)]];galleries[c]=vectors[chosen];queries.extend(i for i in idx if i not in chosen)
        def scores(i):return {c:fused_template_score(galleries[c],vectors[i],dimensions,weights,mode) for c in cows}
        for i in queries:
            s=scores(i);pred=max(cows,key=lambda c:s[c]);true=rows[i]['animal_id']
            closed.append({'fold':fold,'source_path':rows[i]['image_path'],'animal_id':true,'predicted_id':pred,'similarity':s[pred],'correct':pred==true})
            if true in known:
                pred=max(known,key=lambda c:s[c]);kq.append({'score':s[pred],'correct':pred==true,'animal_id':true})
        for c in sorted(unknown):
            for i in indices[c]:s=scores(i);uq.append({'score':max(s[k] for k in known),'animal_id':c})
    positive=[];negative=[]
    matrix=sum(w*(b@b.T) for w,b in zip(weights,branches))
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):(positive if rows[i]['animal_id']==rows[j]['animal_id'] else negative).append(float(matrix[i,j]))
    return {'cow_count':len(cows),'image_count':len(rows),'closed_rank1':float(np.mean([q['correct'] for q in closed])),
            'closed_queries':closed,'known_gallery_cows':len(known),'unknown_cows':len(unknown),'known_queries':kq,'unknown_queries':uq,'positive_pairs':positive,'negative_pairs':negative}
