"""Integrity checks for enrollment transactions, versioning and matching protocol."""
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.registry import CowRegistry
from src.biometrics import ArcFace,GeMPool,rejection_threshold,retrieval_protocol,normalize
from train_identity_reliability import supervised_contrastive
from src.capture_quality import inspect_capture
from PIL import Image

class IntegrityTests(unittest.TestCase):
    def test_duplicate_photo_rolls_back_new_cow(self):
        with tempfile.TemporaryDirectory() as temp:
            registry=CowRegistry(Path(temp)/'registry.sqlite3','model-A')
            v=np.zeros((2,256),dtype=np.float32);v[:,0]=1
            registry.enroll('A',{},v,['a','b'],['hash-a','hash-b'])
            with self.assertRaises(sqlite3.IntegrityError):registry.enroll('B',{},v,['b','c'],['hash-b','hash-c'])
            self.assertEqual(registry.count(),1)
            self.assertEqual(registry.candidates(v[0])[0]['cow_id'],'A')
            registry.close()
    def test_model_versions_cannot_mix(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'registry.sqlite3';r=CowRegistry(p,'model-A');r.close()
            with self.assertRaises(ValueError):CowRegistry(p,'model-B')
    def test_matching_mode_is_versioned(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'registry.sqlite3';r=CowRegistry(p,'model-A');r.close()
            with self.assertRaises(ValueError):CowRegistry(p,'model-A',matching_mode='hybrid')
    def test_runtime_hybrid_matches_protocol_formula(self):
        with tempfile.TemporaryDirectory() as temp:
            r=CowRegistry(Path(temp)/'registry.sqlite3','model-A',matching_mode='hybrid')
            v=np.zeros((2,256),dtype=np.float32);v[0,0]=1;v[1,1]=1;q=v[0]
            r.enroll('A',{},v,['a','b'],['a','b'])
            expected=0.5*(float(normalize(v.mean(axis=0))@q)+float(np.max(v@q)))
            self.assertAlmostEqual(r.candidates(q)[0]['similarity'],expected,places=6);r.close()
    def test_arcface_gradients_are_finite(self):
        head=ArcFace(3);x=torch.nn.functional.normalize(torch.randn(6,256),dim=1).requires_grad_();labels=torch.tensor([0,0,1,1,2,2])
        loss=torch.nn.functional.cross_entropy(head(x,labels),labels);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(x.grad).all())
    def test_query_exclusion_and_threshold_ties(self):
        # Every photo is a distinct orthogonal vector. Leaking query pixels into
        # its enrollment prototype would produce a nonzero similarity.
        e=np.eye(9,dtype=np.float32);rows=[{'animal_id':f'cow_{i//3+1}','image_path':str(i)} for i in range(9)]
        result=retrieval_protocol(e,rows)
        self.assertTrue(all(r['similarity']==0 for r in result['closed_queries']))
        threshold=rejection_threshold([0.7]*100)
        self.assertGreater(threshold,0.7)
        for mode in ('max_template','hybrid'):
            result=retrieval_protocol(e,rows,mode)
            self.assertTrue(all(r['similarity']==0 for r in result['closed_queries']))
    def test_contrastive_loss_and_gem_remain_finite(self):
        x=torch.nn.functional.normalize(torch.randn(6,256),dim=1).requires_grad_();labels=torch.tensor([0,0,0,1,1,1])
        loss=supervised_contrastive(x,labels);loss.backward();self.assertTrue(torch.isfinite(x.grad).all())
        with self.assertRaises(ValueError):supervised_contrastive(x,torch.arange(6))
        pooled=GeMPool()(torch.full((2,3,7,7),100.0,dtype=torch.float16))
        self.assertTrue(torch.isfinite(pooled).all())
    def test_flat_small_capture_is_flagged(self):
        q=inspect_capture(Image.new('RGB',(40,40),'gray'),{'min_short_side':96,'sharpness_review_below':1,'contrast_review_below':20})
        self.assertIn('muzzle_resolution_low',q['flags']);self.assertIn('possible_blur_or_low_texture',q['flags'])

if __name__=='__main__':unittest.main()
