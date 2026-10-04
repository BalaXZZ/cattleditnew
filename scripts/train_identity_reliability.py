"""Validation-only metric-learning experiments. Never load test rows or images."""
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from src.biometrics import MuzzleEncoder,ArcFace,retrieval_protocol,rejection_threshold,summarize_protocol
from train_muzzle_embeddings import CropDataset,encode,save_checkpoint

CONFIGS=[
    {'name':'balanced_avg','pooling':'avg','initial':'imagenet','encoder_lr':0.00001},
    {'name':'balanced_gem','pooling':'gem','initial':'imagenet','encoder_lr':0.00001},
    {'name':'refine_baseline','pooling':'avg','initial':'baseline','encoder_lr':0.000005},
    {'name':'frozen_gem','pooling':'gem','initial':'imagenet','encoder_lr':0.0},
]

def supervised_contrastive(vectors,labels,temperature=0.1):
    """Every anchor has other real photos of its cow in the balanced batch."""
    vectors=F.normalize(vectors.float(),dim=1);logits=(vectors@vectors.T)/temperature
    diagonal=torch.eye(len(labels),device=labels.device,dtype=torch.bool)
    positives=(labels[:,None]==labels[None,:])&~diagonal
    logits=logits-logits.max(dim=1,keepdim=True).values.detach()
    denominators=torch.logsumexp(logits.masked_fill(diagonal,float('-inf')),dim=1)
    log_prob=logits-denominators[:,None]
    counts=positives.sum(dim=1);valid=counts>0
    if not valid.any():raise ValueError('Balanced batch has no positive pairs')
    return -((log_prob.masked_fill(~positives,0).sum(dim=1)/counts.clamp_min(1))[valid]).mean()

def evaluate(vectors,rows,mode):
    protocol=retrieval_protocol(vectors,rows,mode)
    pt=rejection_threshold(protocol['negative_pairs'])
    it=rejection_threshold([q['score'] for q in protocol['unknown_queries']])
    wrong=[q['score'] for q in protocol['known_queries'] if not q['correct']]
    if wrong:it=max(it,float(np.nextafter(max(wrong),np.inf)))
    summary=summarize_protocol(protocol,pt,it)
    # Prefer both reliable accepted matches and closed-set discrimination.
    score=(summary['closed_rank1']+summary['open_identification']['known_correct_acceptance'])/2
    return score,summary,pt,it

def main():
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise SystemExit('CUDA unavailable')
    out=ROOT/'outputs/reliability/experiments';out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'data/embedding/manifest.json').read_text())
    # Explicitly retain only development partitions. Test data is never decoded.
    train_rows=[r for r in manifest['rows'] if r['split']=='train'];val_rows=[r for r in manifest['rows'] if r['split']=='val']
    classes={c:i for i,c in enumerate(manifest['train_classes'])}
    grouped={c:[i for i,r in enumerate(train_rows) if r['animal_id']==c] for c in classes}
    train=CropDataset(train_rows,classes,True);val=DataLoader(CropDataset(val_rows),batch_size=16)
    baseline_vectors=np.load(ROOT/'outputs/reliability/baseline_val.npz')['embeddings']
    baseline_score,baseline_summary,_,_=evaluate(baseline_vectors,val_rows,'centroid')
    results=[{'name':'baseline','score':baseline_score,'validation':baseline_summary,'mode':'centroid'}]
    (out/'experiment_plan.json').write_text(json.dumps({'configs':CONFIGS,'seed':42,'max_epochs':40,'early_stop':10,'positive_photos_per_cow':3,'cows_per_batch':8,'selection':'Average validation closed rank-1 and known correct acceptance at conservative unknown FAR <=1%; zero known wrong accepts in validation.','test_access':False},indent=2))
    for config in CONFIGS:
        run=out/config['name']
        if (run/'validation.json').exists():
            results.append(json.loads((run/'validation.json').read_text()));continue
        if run.exists():raise SystemExit('Incomplete run exists; preserve it and choose a new experiment name.')
        run.mkdir();random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        model=MuzzleEncoder(ROOT/'models/resnet18-f37072fd.pth',config['pooling']).cuda()
        if config['initial']=='baseline':model.load_state_dict(torch.load(ROOT/'outputs/embeddings/resnet18_arcface/best.pt',map_location='cpu',weights_only=True)['encoder'])
        for name,p in model.backbone.named_parameters():p.requires_grad=config['encoder_lr']>0 and name.startswith(('layer3','layer4'))
        head=ArcFace(len(classes)).cuda()
        params=[{'params':model.projection.parameters(),'lr':0.0003},{'params':head.parameters(),'lr':0.0005}]
        encoder_params=[p for p in model.backbone.parameters() if p.requires_grad]
        if encoder_params:params.append({'params':encoder_params,'lr':config['encoder_lr']})
        optimizer=torch.optim.AdamW(params,weight_decay=0.001);scaler=torch.amp.GradScaler('cuda')
        best=-1;best_epoch=0;history=[];begin=time.monotonic()
        for epoch in range(1,41):
            model.train();head.train()
            for module in model.backbone.modules():
                if isinstance(module,nn.BatchNorm2d):module.eval()
            factor=float(0.5*(1+np.cos(np.pi*(epoch-1)/40)))
            for j,g in enumerate(optimizer.param_groups):g['lr']=[0.0003,0.0005,0 if epoch<=2 else config['encoder_lr']][j]*factor
            cows=list(classes);random.shuffle(cows);total=0;steps=0
            for offset in range(0,len(cows),8):
                selected=cows[offset:offset+8];indices=[]
                if len(selected)<2:continue
                for c in selected:indices.extend(random.sample(grouped[c],3))
                batch=[train[i] for i in indices];images=torch.stack([b[0] for b in batch]).cuda();labels=torch.tensor([b[1] for b in batch],device='cuda')
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda'):vectors=model(images)
                metric_loss=supervised_contrastive(vectors,labels)
                arc_weight=0.05 if epoch<=3 else 0.25
                margin=0.2*min(1,max(0,(epoch-3)/5))
                loss=metric_loss+arc_weight*F.cross_entropy(head(vectors.float(),labels,margin),labels)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),5)
                scaler.step(optimizer);scaler.update();total+=loss.item();steps+=1
            vectors=encode(model,val,'cuda');options=[]
            for mode in ('centroid','max_template','hybrid'):
                score,summary,pt,it=evaluate(vectors,val_rows,mode);options.append((score,mode,summary,pt,it))
            score,mode,summary,pt,it=max(options,key=lambda item:item[0]);improved=score>best
            if improved:
                best=score;best_epoch=epoch
                cp={'encoder':model.state_dict(),'architecture':'resnet18_metric_256','pooling':config['pooling'],'dimension':256,'input_size':224,
                    'epoch':epoch,'seed':42,'config':config,'matching_mode':mode,'validation':summary,'selection_score':score,'train_classes':manifest['train_classes']}
                save_checkpoint(run/'best.pt',cp);np.savez(run/'validation_embeddings.npz',embeddings=vectors)
            row={'epoch':epoch,'loss':total/steps,'selection_score':score,'rank1':summary['closed_rank1'],'known_correct_acceptance':summary['open_identification']['known_correct_acceptance'],'matching_mode':mode,'best_epoch':best_epoch,'seconds':time.monotonic()-begin};history.append(row)
            with (run/'history.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
            print(config['name'],json.dumps(row),flush=True)
            if epoch>=15 and epoch-best_epoch>=10:break
        cp=torch.load(run/'best.pt',map_location='cpu',weights_only=True);vectors=np.load(run/'validation_embeddings.npz')['embeddings']
        _,summary,pt,it=evaluate(vectors,val_rows,cp['matching_mode'])
        calibration={'model_sha256':hashlib.sha256((run/'best.pt').read_bytes()).hexdigest(),'matching_mode':cp['matching_mode'],'pair_threshold':pt,'identification_threshold':it,'validation':summary,'target_empirical_false_acceptance':0.01,'warning':'Validation-only development results after model selection; small dependent trials, 13-cow calibration gallery. Independent new evaluation required.'}
        (run/'calibration.json').write_text(json.dumps(calibration,indent=2))
        result={'name':config['name'],'score':cp['selection_score'],'best_epoch':cp['epoch'],'mode':cp['matching_mode'],'epochs_completed':len(history),'validation':summary}
        (run/'validation.json').write_text(json.dumps(result,indent=2));results.append(result)
        (out/'comparison.json').write_text(json.dumps(results,indent=2))
        del model,head,optimizer;torch.cuda.empty_cache()
    eligible=[r for r in results if r['validation']['closed_rank1']>=baseline_summary['closed_rank1'] and r['validation']['open_identification']['known_correct_acceptance']>=baseline_summary['open_identification']['known_correct_acceptance']]
    winner=max(eligible,key=lambda r:r['score'])
    (out/'selection.json').write_text(json.dumps({'winner':winner,'baseline':results[0],'test_access':False,'note':'Selection uses validation cows only. No final accuracy claim; fresh independent evaluation remains required.'},indent=2))
    print('Selected:',json.dumps(winner),flush=True)

if __name__=='__main__':main()
