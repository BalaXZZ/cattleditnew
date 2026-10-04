"""Local cow IDs and model-versioned biometric templates; no automatic enrollment."""
import json
import sqlite3
from pathlib import Path
import numpy as np
from src.biometrics import normalize

class CowRegistry:
    def __init__(self,path,model_hash,dimension=256,matching_mode='centroid',branch_dimensions=None,branch_weights=None):
        if matching_mode not in ('centroid','max_template','hybrid'):raise ValueError('Unknown matching mode')
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.connection=sqlite3.connect(self.path)
        self.connection.execute('PRAGMA foreign_keys=ON')
        self.connection.executescript('''
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cows (cow_id TEXT PRIMARY KEY, details TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS templates (cow_id TEXT NOT NULL REFERENCES cows(cow_id), image_hash TEXT NOT NULL UNIQUE, source_path TEXT NOT NULL, embedding BLOB NOT NULL, input_kind TEXT NOT NULL);
        ''')
        metadata=dict(self.connection.execute('SELECT key,value FROM metadata'))
        self.branch_dimensions=branch_dimensions or [dimension];self.branch_weights=branch_weights or [1.0]
        if sum(self.branch_dimensions)!=dimension or len(self.branch_dimensions)!=len(self.branch_weights) or min(self.branch_weights)<0 or not np.isclose(sum(self.branch_weights),1):
            self.connection.close();raise ValueError('Invalid stored branch specification')
        recorded_dims=json.loads(metadata.get('branch_dimensions',json.dumps([dimension])))
        recorded_weights=json.loads(metadata.get('branch_weights','[1.0]'))
        if metadata and (metadata.get('model_hash')!=model_hash or metadata.get('dimension')!=str(dimension) or metadata.get('matching_mode','centroid')!=matching_mode or recorded_dims!=self.branch_dimensions or recorded_weights!=self.branch_weights):
            self.connection.close();raise ValueError('Registry uses another model version. Re-embed stored source photos; do not mix embedding versions.')
        if not metadata:
            with self.connection:self.connection.executemany('INSERT INTO metadata VALUES (?,?)',[('model_hash',model_hash),('dimension',str(dimension)),('matching_mode',matching_mode),('branch_dimensions',json.dumps(self.branch_dimensions)),('branch_weights',json.dumps(self.branch_weights))])
        if 'input_kind' not in {r[1] for r in self.connection.execute('PRAGMA table_info(templates)')}:
            with self.connection:self.connection.execute("ALTER TABLE templates ADD COLUMN input_kind TEXT NOT NULL DEFAULT 'unknown'")
        self.dimension=dimension
        self.matching_mode=matching_mode
    def close(self):self.connection.close()
    def candidates(self,embedding,limit=3):
        grouped={}
        for cow,blob in self.connection.execute('SELECT cow_id,embedding FROM templates'):
            vector=np.frombuffer(blob,dtype=np.float32)
            if len(vector)!=self.dimension:raise ValueError('Invalid stored embedding dimension')
            grouped.setdefault(cow,[]).append(vector)
        if not grouped:return []
        scores=[]
        for cow,vectors in grouped.items():
            from src.hybrid_biometrics import fused_template_score
            score=fused_template_score(np.stack(vectors),embedding,self.branch_dimensions,self.branch_weights,self.matching_mode)
            details=json.loads(self.connection.execute('SELECT details FROM cows WHERE cow_id=?',(cow,)).fetchone()[0])
            scores.append({'cow_id':cow,'similarity':score,'details':details})
        return sorted(scores,key=lambda r:r['similarity'],reverse=True)[:limit]
    def count(self):return self.connection.execute('SELECT COUNT(*) FROM cows').fetchone()[0]
    def enroll(self,cow_id,details,vectors,sources,hashes,input_kind='muzzle_crop'):
        if not cow_id.strip() or len(cow_id)>128:raise ValueError('Invalid cow ID')
        if not isinstance(details,dict):raise ValueError('Details must be a JSON object')
        if input_kind not in ('muzzle_crop','full_image'):raise ValueError('Invalid input kind')
        if len(vectors)<1 or len(set(hashes))!=len(hashes):raise ValueError('Enrollment requires distinct photos')
        if not(len(vectors)==len(sources)==len(hashes)):raise ValueError('Inconsistent template inputs')
        vectors=np.asarray(vectors,dtype=np.float32)
        if vectors.shape[1]!=self.dimension or not np.isfinite(vectors).all():raise ValueError('Invalid vectors')
        with self.connection:
            self.connection.execute('INSERT INTO cows(cow_id,details) VALUES (?,?)',(cow_id,json.dumps(details)))
            self.connection.executemany('INSERT INTO templates(cow_id,image_hash,source_path,embedding,input_kind) VALUES (?,?,?,?,?)',[(cow_id,h,str(p),v.astype(np.float32).tobytes(),input_kind) for v,p,h in zip(vectors,sources,hashes)])
    def list_cows(self):
        return [{'cow_id':cow,'details':json.loads(details),'photo_count':n} for cow,details,n in self.connection.execute('SELECT c.cow_id,c.details,COUNT(t.image_hash) FROM cows c LEFT JOIN templates t ON t.cow_id=c.cow_id GROUP BY c.cow_id ORDER BY c.cow_id')]
