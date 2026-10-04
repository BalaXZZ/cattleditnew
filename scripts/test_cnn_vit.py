"""Branch fusion, versioning and offline ViT checkpoint checks."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.registry import CowRegistry
from src.biometrics import load_encoder
from src.hybrid_biometrics import DinoMuzzleEncoder,fusion_protocol,fused_template_score,fused_pair_score
from src.identity_system import IdentitySystem,bundle_hash

class FusionTests(unittest.TestCase):
    def test_fusion_respects_each_branch_weight(self):
        with tempfile.TemporaryDirectory() as temp:
            r=CowRegistry(Path(temp)/'r.sqlite3','bundle',dimension=4,matching_mode='hybrid',branch_dimensions=[2,2],branch_weights=[0.75,0.25])
            a=np.array([[1,0,0,1],[1,0,0,1]],dtype=np.float32);b=np.array([[0,1,1,0],[0,1,1,0]],dtype=np.float32)
            r.enroll('A',{},a,['a1','a2'],['a1','a2']);r.enroll('B',{},b,['b1','b2'],['b1','b2'])
            q=np.array([1,0,1,0],dtype=np.float32);c=r.candidates(q)
            self.assertEqual(c[0]['cow_id'],'A');self.assertAlmostEqual(c[0]['similarity'],0.75);r.close()
            with self.assertRaises(ValueError):CowRegistry(Path(temp)/'r.sqlite3','bundle',dimension=4,matching_mode='hybrid',branch_dimensions=[2,2],branch_weights=[0.25,0.75])
    def test_pair_score_is_not_unweighted_concatenation(self):
        a=np.array([1,0,1,0],dtype=np.float32);b=np.array([1,0,0,1],dtype=np.float32)
        self.assertAlmostEqual(fused_pair_score(a,b,[2,2],[0.25,0.75]),0.25)
    def test_query_does_not_leak_into_either_branch_gallery(self):
        e=np.eye(9,dtype=np.float32);rows=[{'animal_id':f'cow_{i//3+1}','image_path':str(i)} for i in range(9)]
        result=fusion_protocol([e,e],rows,[0.5,0.5])
        self.assertTrue(all(q['similarity']==0 for q in result['closed_queries']))
    def test_tiny_vit_checkpoint_loads_offline(self):
        config={'hidden_size':24,'num_hidden_layers':1,'num_attention_heads':3,'intermediate_size':48,'image_size':28,'patch_size':14}
        model=DinoMuzzleEncoder(config=config,feature_mode='cls_patch',projection_dim=8).eval()
        x=torch.randn(2,3,28,28)
        with torch.no_grad():expected=model(x)
        self.assertTrue(torch.isfinite(expected).all());self.assertTrue(torch.allclose(expected.norm(dim=1),torch.ones(2),atol=1e-5))
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'vit.pt';torch.save({'encoder':model.state_dict(),'architecture':'dinov2_metric_256','vit_config':model.backbone.config.to_dict(),'feature_mode':'cls_patch','projection_dim':8,'dimension':8},p)
            restored,_=load_encoder(p)
            with torch.no_grad():actual=restored(x)
            self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
    def test_bundle_hash_binds_weights_and_fusion_rule(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'cnn.pt').write_bytes(b'cnn');(p/'vit.pt').write_bytes(b'vit')
            desc={'format_version':1,'branches':[{'name':'cnn','checkpoint':'cnn.pt','dimension':2},{'name':'vit','checkpoint':'vit.pt','dimension':2}],'weights':[0.5,0.5],'matching_mode':'hybrid'}
            digest=bundle_hash(p,desc);(p/'system.json').write_text(json.dumps(desc));(p/'calibration.json').write_text(json.dumps({'model_sha256':digest,'matching_mode':'hybrid'}))
            s=IdentitySystem(p,load_models=False);self.assertEqual(s.dimension,4)
            (p/'vit.pt').write_bytes(b'changed weights')
            with self.assertRaises(ValueError):IdentitySystem(p,load_models=False)

if __name__=='__main__':unittest.main()
