"""Package the selected architecture, binding encoder weights to calibration."""
import json
import os
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.identity_system import bundle_hash

def main():
    selection=json.loads((ROOT/'outputs/cnn_vit/selection.json').read_text());winner=selection['winner']
    if winner['vit_weight']==0:raise SystemExit('ViT did not improve selection; retain active CNN.')
    destination=ROOT/'outputs/embeddings/cnn_vit_v3'
    if destination.exists():raise SystemExit('Bundle exists; refusing overwrite.')
    destination.mkdir();shutil.copy2(ROOT/'outputs/embeddings/reliability_v2/best.pt',destination/'cnn.pt')
    source=ROOT/'outputs/cnn_vit'/winner['vit_candidate'];shutil.copy2(source/'best.pt',destination/'vit.pt')
    import torch
    cp=torch.load(source/'best.pt',map_location='cpu',weights_only=True)
    descriptor={'format_version':1,'preprocessing':'rgb_squarepad224_imagenet_v1',
                'branches':[{'name':'cnn','checkpoint':'cnn.pt','dimension':256},{'name':'vit','checkpoint':'vit.pt','dimension':cp['dimension']}],
                'weights':[winner['cnn_weight'],winner['vit_weight']],'matching_mode':'hybrid'}
    (destination/'system.json').write_text(json.dumps(descriptor,indent=2));digest=bundle_hash(destination,descriptor)
    calibration={'model_sha256':digest,'matching_mode':'hybrid','pair_threshold':winner['pair_threshold'],
                 'identification_threshold':winner['identification_threshold'],'validation':winner['validation'],
                 'warning':'Development validation used for selection, not fresh test accuracy. Thresholds calibrated with 13 enrolled cows; no field or statewide error guarantee.'}
    (destination/'calibration.json').write_text(json.dumps(calibration,indent=2))
    shutil.copy2(ROOT/'outputs/embeddings/reliability_v2/capture_policy.json',destination/'capture_policy.json')
    (destination/'development_report.json').write_text(json.dumps({'selection':selection,'vit_best_epoch':cp['epoch'],'model_hash':digest},indent=2))
    active=ROOT/'outputs/embeddings/active_model.json';backup=active.with_name('active_model_before_cnn_vit.json')
    if not backup.exists():shutil.copy2(active,backup)
    temp=active.with_suffix('.tmp');temp.write_text(json.dumps({'model_dir':destination.relative_to(ROOT).as_posix(),'model_sha256':digest,'status':'development_only_requires_review'},indent=2));os.replace(temp,active)
    print('Active CNN/ViT bundle:',destination,'hash:',digest)

if __name__=='__main__':main()
