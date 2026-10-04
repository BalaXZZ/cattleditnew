"""Produce a readable comparison report after the controlled experiment finishes."""
import json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiment_mobilenet_cosface import OUT
DEST=ROOT/'outputs'

def pct(x):return f'{100*x:.1f}%'
def row(name,m):
 o=m['open_identification'];p=m['pair_verification']
 return f"| {name} | {pct(m['closed_rank1'])} | {pct(o['known_correct_acceptance'])} | {pct(o['unknown_false_acceptance'])} | {pct(p['balanced_accuracy'])} |"

def main():
 result=json.loads((OUT/'comparison.json').read_text());DEST.mkdir(exist_ok=True)
 lines=['# MobileNetV3-L + CosFace experiment results','',
 'All three candidates were trained and compared with the active CNN + ViT baseline. The active model, its calibration and the enrolled registry were not changed.','',
 '## Development validation','',
 '| Model | Correct top match | Known cow correctly accepted | Unknown cow falsely accepted | Balanced pair accuracy |',
 '|---|---:|---:|---:|---:|']
 lines.append(row('Current CNN + ViT',result['baseline']['validation']['full']))
 for candidate in result['candidates']:lines.append(row(candidate['config']['name'],candidate['validation']['full']))
 lines += ['', 'Checkpoint and candidate selection used validation only. Each candidate has its own validation-calibrated thresholds. The known-ID acceptance rule also excludes wrong known matches on validation. Pair accuracy uses a separately selected balanced operating point, which is not the application acceptance threshold.','',
 '## Partial-view development validation','',
 '| Model | Full muzzle top match | Left half top match | Right half top match |',
 '|---|---:|---:|---:|']
 for name,views in [('CNN + ViT',result['baseline']['validation'])]+[(r['config']['name'],r['validation']) for r in result['candidates']]:
  lines.append('| '+name+' | '+' | '.join(pct(views[v]['closed_rank1']) for v in ['full','left_half','right_half'])+' |')
 lines += ['', 'These are synthetic half crops from existing muzzle photos, not independently captured oblique views. Enrollment uses full muzzles from other photographs. Query source photos are excluded from the enrolled gallery in every fold. The original quality-filtered images remain fixed; no hard query was removed.','',
 '## Held-out comparison after validation selection','',
 f"Validation selected `{result['selection']['candidate']}` before test decoding. Eligible to replace the baseline on both development metrics: **{result['selection']['eligible_vs_baseline']}**.", '',
 '| Model / query | Correct top match | Known cow correctly accepted | Unknown cow falsely accepted | Balanced pair accuracy |',
 '|---|---:|---:|---:|---:|']
 for name,views in result['held_out_test'].items():
  for view,m in views.items():lines.append(row(name+' / '+view,m))
 base=result['held_out_test']['baseline']['full'];candidate=result['held_out_test']['candidate']['full']
 lines += ['', 'Thresholds from full-image validation were applied unchanged to every held-out query condition. The test was not used for candidate selection or threshold tuning.','',
 f"Held-out full-image rank-1 change: {(candidate['closed_rank1']-base['closed_rank1'])*100:+.1f} percentage points. Candidate exceeds 90% held-out rank-1: **{candidate['closed_rank1']>0.9}**.", '',
 '## Training recipe and scope','',
 '- Official torchvision ImageNet MobileNetV3-L pretrained weights, downloaded locally with SHA prefix verification.',
 '- 256-dimensional normalized embeddings; CosFace margin 0.35, scale 30, five-epoch margin warmup.',
 '- Same train/validation/test cow split as the existing project; no enrollment registry data used.',
 '- Seed 42, up to 35 epochs, balanced batches of four cows with three actual photographs per cow.',
 '- AdamW, small learning rates, final feature blocks fine-tuned; frozen BatchNorm statistics. This is a transfer-learning adaptation, not an exact replication of the paper\'s full SGD recipe.',
 '- Matched 224- and 384-pixel experiments differ only in input resolution. Original native crop pixels are used for both.',
 '- Partial candidate uses a 35% chance of a randomly selected left/right half during training; other augmentation and training settings remain fixed.',
 '- Early stopping after 10 epochs without improvement, at least 15 epochs; selected score averages rank-1 retrieval and correctly accepted known cows.',
 '- Keypoint alignment was NOT evaluated: the current labels contain bounding boxes, not verified nostril landmarks. No anatomical keypoints were fabricated.', '',
 '## Interpretation','',
 result['warning'], '',
 'The held-out partition contains cows not used for training or current validation selection, but it was used in historical baseline work. It is not a fresh field test. Repeated gallery folds and overlapping pairs are dependent. These results cannot establish statewide false-match rates. A single training seed measures one experiment, not statistical superiority.','',
 'The paper\'s open-set metric is 1:1 pair verification. Application lookup is 1:N identification with unknown rejection. A model may exceed 90% balanced pair accuracy while still failing to exceed 90% correct gallery identification.', '',
 'No model was activated and no existing registry embeddings were regenerated. If a future encoder is adopted, preserve cow IDs/details while migrating all saved source photos to the new model and calibrating matching thresholds.', '',
 'Experiment code: scripts/experiment_mobilenet_cosface.py. Reproduce protocol checks: scripts/test_mobilenet_experiment.py. Full machine-readable metrics accompany this report.','',
 'Research motivation: https://doi.org/10.1016/j.patcog.2026.114417']
 (DEST/'mobilenet_cosface_results.md').write_text('\n'.join(lines),encoding='utf-8')
 (DEST/'mobilenet_cosface_comparison.json').write_text(json.dumps(result,indent=2))
 with zipfile.ZipFile(DEST/'mobilenet_cosface_experimental_checkpoints.zip','w',zipfile.ZIP_STORED) as archive:
  for c in result['candidates']:
   folder=OUT/c['config']['name']
   for filename in ['best.pt','result.json','history.json']:archive.write(folder/filename,c['config']['name']+'/'+filename)
  archive.write(ROOT/'scripts/experiment_mobilenet_cosface.py','experiment_mobilenet_cosface.py')
 print(str(DEST/'mobilenet_cosface_results.md'))
 print(json.dumps({'selection':result['selection'],'baseline_test_rank1':base['closed_rank1'],'candidate_test_rank1':candidate['closed_rank1']}))

if __name__=='__main__':main()
