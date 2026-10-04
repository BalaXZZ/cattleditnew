"""Train a DINOv2 branch and compare score fusion on validation cows only."""
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.biometrics import ArcFace,load_encoder,rejection_threshold,summarize_protocol,retrieval_protocol
from src.hybrid_biometrics import DinoMuzzleEncoder,fusion_protocol
from train_muzzle_embeddings import CropDataset,encode,save_checkpoint
from train_identity_reliability import supervised_contrastive

WEIGHTS=[0.0,0.25,0.5,0.75,1.0]

def summarize(result):
    pt=rejection_threshold(result['negative_pairs']);it=rejection_threshold([q['score'] for q in result['unknown_queries']])
    wrong=[q['score'] for q in result['known_queries'] if not q['correct']]
    if wrong:it=max(it,float(np.nextafter(max(wrong),np.inf)))
    summary=summarize_protocol(result,pt,it)
    score=(summary['closed_rank1']+summary['open_identification']['known_correct_acceptance'])/2
    return score,summary,pt,it

def compare(cnn,vit,rows):
    options=[]
    for weight in WEIGHTS:
        # Keep hybrid template scoring fixed to the already-selected CNN rule.
        result=fusion_protocol([cnn,vit],rows,[weight,1-weight],'hybrid')
        score,summary,pt,it=summarize(result)
        options.append({'cnn_weight':weight,'vit_weight':1-weight,'score':score,'validation':summary,'pair_threshold':pt,'identification_threshold':it})
    return options

def metadata(model,name):
    return {'architecture':'dinov2_metric_256' if isinstance(model.projection,nn.Linear) else 'dinov2_frozen_features',
            'feature_mode':model.feature_mode,'projection_dim':model.dimension if isinstance(model.projection,nn.Linear) else None,
            'dimension':model.dimension,'vit_config':model.backbone.config.to_dict(),'input_size':224,'seed':42,'candidate':name}

def main():
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise SystemExit('CUDA required')
    out=ROOT/'outputs/cnn_vit';out.mkdir(exist_ok=True)
    if (out/'comparison.json').exists():raise SystemExit('Completed comparison exists; refusing overwrite.')
    manifest=json.loads((ROOT/'data/embedding/manifest.json').read_text());train_rows=[r for r in manifest['rows'] if r['split']=='train'];val_rows=[r for r in manifest['rows'] if r['split']=='val']
    classes={c:i for i,c in enumerate(manifest['train_classes'])};grouped={c:[i for i,r in enumerate(train_rows) if r['animal_id']==c] for c in classes}
    train=CropDataset(train_rows,classes,True);val=DataLoader(CropDataset(val_rows),batch_size=8)
    cnn_path=ROOT/'outputs/embeddings/reliability_v2/best.pt';cnn,_=load_encoder(cnn_path,'cuda');cnn_vectors=encode(cnn,val,'cuda');del cnn;torch.cuda.empty_cache()
    np.savez(out/'cnn_validation.npz',embeddings=cnn_vectors)
    plan={'cnn_checkpoint':str(cnn_path),'vit_pretrained':'facebook/dinov2-small','candidates':['frozen_cls','frozen_cls_patch','trained_head','last_two_blocks'],
          'cnn_weights':WEIGHTS,'gallery_scoring':'hybrid','selection':'Mean validation closed rank-1 and correct known acceptance at unknown empirical FAR <=1%, zero wrong known accepts in validation. Require no regression on CNN rank-1 and correct known acceptance.','seed':42,'max_epochs':30,'staged_unfreeze_epoch':6,'early_stop_patience':8,'test_access':False}
    (out/'experiment_plan.json').write_text(json.dumps(plan,indent=2));records=[]
    baseline=summarize(retrieval_protocol(cnn_vectors,val_rows,'hybrid'))[1]
    for name in plan['candidates']:
        run=out/name;run.mkdir(exist_ok=True)
        if (run/'result.json').exists():records.extend(json.loads((run/'result.json').read_text()));continue
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        frozen=name.startswith('frozen');feature_mode='cls' if name=='frozen_cls' else 'cls_patch'
        model=DinoMuzzleEncoder(pretrained_path=str(ROOT/'models/dinov2-small'),feature_mode=feature_mode,projection_dim=None if frozen else 256).cuda()
        for p in model.backbone.parameters():p.requires_grad=False
        if frozen:
            vectors=encode(model,val,'cuda');options=compare(cnn_vectors,vectors,val_rows)
            save_checkpoint(run/'best.pt',{'encoder':model.state_dict(),**metadata(model,name),'epoch':0})
            np.savez(run/'validation_embeddings.npz',embeddings=vectors)
            print(name,json.dumps(max(options,key=lambda x:x['score'])),flush=True)
        else:
            head=ArcFace(len(classes)).cuda()
            if name=='last_two_blocks':
                for layer in model.backbone.encoder.layer[-2:]:
                    for p in layer.parameters():p.requires_grad=True
            params=[{'params':model.projection.parameters(),'lr':0.0003},{'params':head.parameters(),'lr':0.0005}]
            backbone_params=[p for p in model.backbone.parameters() if p.requires_grad]
            if backbone_params:params.append({'params':backbone_params,'lr':0.000003})
            optimizer=torch.optim.AdamW(params,weight_decay=0.001);scaler=torch.amp.GradScaler('cuda');best=-1;best_epoch=0;history=[];begin=time.monotonic()
            for epoch in range(1,31):
                model.train();head.train();model.backbone.eval()
                if name=='last_two_blocks' and epoch>=6:
                    for layer in model.backbone.encoder.layer[-2:]:layer.train()
                factor=float(0.5*(1+np.cos(np.pi*(epoch-1)/30)))
                for j,g in enumerate(optimizer.param_groups):g['lr']=[0.0003,0.0005,0 if epoch<6 else 0.000003][j]*factor
                cows=list(classes);random.shuffle(cows);losses=[]
                for offset in range(0,len(cows),4):
                    selected=cows[offset:offset+4]
                    if len(selected)<2:continue
                    idx=[i for c in selected for i in random.sample(grouped[c],3)]
                    batch=[train[i] for i in idx];images=torch.stack([b[0] for b in batch]).cuda();labels=torch.tensor([b[1] for b in batch],device='cuda')
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda'):vectors=model(images)
                    loss=supervised_contrastive(vectors,labels)+(0.05 if epoch<4 else 0.25)*F.cross_entropy(head(vectors.float(),labels,0.2*min(1,max(0,(epoch-3)/5))),labels)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite ViT training loss')
                    scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),5)
                    scaler.step(optimizer);scaler.update();losses.append(loss.item())
                vectors=encode(model,val,'cuda');options=compare(cnn_vectors,vectors,val_rows)
                # Checkpoint selection cannot sacrifice either baseline metric.
                eligible=[o for o in options if o['cnn_weight']<1 and o['validation']['closed_rank1']>=baseline['closed_rank1'] and o['validation']['open_identification']['known_correct_acceptance']>=baseline['open_identification']['known_correct_acceptance']]
                candidate=max(eligible or [o for o in options if o['cnn_weight']<1],key=lambda x:x['score'])
                score=candidate['score']
                if score>best:
                    best=score;best_epoch=epoch;save_checkpoint(run/'best.pt',{'encoder':model.state_dict(),**metadata(model,name),'epoch':epoch})
                    np.savez(run/'validation_embeddings.npz',embeddings=vectors)
                row={'epoch':epoch,'loss':float(np.mean(losses)),'selection_score':score,'cnn_weight':candidate['cnn_weight'],'rank1':candidate['validation']['closed_rank1'],'known_correct_acceptance':candidate['validation']['open_identification']['known_correct_acceptance'],'best_epoch':best_epoch,'seconds':time.monotonic()-begin}
                history.append(row);(run/'history.json').write_text(json.dumps(history,indent=2));print(name,json.dumps(row),flush=True)
                if epoch>=12 and epoch-best_epoch>=8:break
            vectors=np.load(run/'validation_embeddings.npz')['embeddings'];options=compare(cnn_vectors,vectors,val_rows)
            del head,optimizer
        results=[{'vit_candidate':name,**o} for o in options];(run/'result.json').write_text(json.dumps(results,indent=2));records.extend(results)
        (out/'progress_comparison.json').write_text(json.dumps(records,indent=2));del model;torch.cuda.empty_cache()
    eligible=[r for r in records if r['validation']['closed_rank1']>=baseline['closed_rank1'] and r['validation']['open_identification']['known_correct_acceptance']>=baseline['open_identification']['known_correct_acceptance']]
    winner=max(eligible,key=lambda r:r['score'])
    (out/'comparison.json').write_text(json.dumps(records,indent=2));(out/'selection.json').write_text(json.dumps({'winner':winner,'cnn_baseline':baseline,'test_access':False,'warning':'Selected validation results, not independent test accuracy or field FAR.'},indent=2))
    print('Selected:',json.dumps(winner),flush=True)

if __name__=='__main__':main()
