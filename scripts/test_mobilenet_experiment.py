import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from experiment_mobilenet_cosface import CosFace,protocol,view_crop
from src.hybrid_biometrics import fusion_protocol

class Checks(unittest.TestCase):
 def test_half_crops_keep_distinct_regions(self):
  image=Image.fromarray(np.tile(np.arange(8,dtype=np.uint8)[None,:,None],(4,1,3)))
  self.assertEqual(np.asarray(view_crop(image,'left_half'))[0,:,0].tolist(),[0,1,2,3])
  self.assertEqual(np.asarray(view_crop(image,'right_half'))[0,:,0].tolist(),[4,5,6,7])
 def test_protocol_matches_existing_fusion(self):
  rows=[{'animal_id':f'cow_{c}','image_path':f'{c}/{i}'} for c in range(1,5) for i in range(3)]
  rng=np.random.default_rng(1);parts=[]
  for width in [8,6]:
   vectors=rng.normal(size=(12,width));vectors/=np.linalg.norm(vectors,axis=1,keepdims=True);parts.append(vectors)
  vectors=np.concatenate(parts,axis=1);p=protocol(vectors,vectors,rows,[8,6],[.5,.5]);old=fusion_protocol(parts,rows,[.5,.5],'hybrid')
  self.assertAlmostEqual(p['closed_rank1'],old['closed_rank1'])
  np.testing.assert_allclose([q['score'] for q in p['known_queries']],[q['score'] for q in old['known_queries']],atol=1e-7)
 def test_gallery_excludes_query_source(self):
  rows=[{'animal_id':f'cow_{c}','image_path':f'{c}/{i}'} for c in range(1,5) for i in range(3)]
  # Each photograph is orthogonal: any accidental source overlap would give score 1.
  vectors=np.eye(12);p=protocol(vectors,vectors,rows,[12],[1.])
  self.assertTrue(all(q['score']==0 for q in p['closed_queries']))
 def test_cosface_penalizes_only_target(self):
  head=CosFace(3);vectors=torch.nn.functional.normalize(torch.randn(2,256),dim=1);labels=torch.tensor([0,2])
  difference=head(vectors,labels,0)-head(vectors,labels,.35)
  expected=torch.nn.functional.one_hot(labels,3)*10.5
  torch.testing.assert_close(difference,expected.float())

if __name__=='__main__':unittest.main()
