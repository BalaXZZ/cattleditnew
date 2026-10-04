"""Plot measured validation and held-out rank-1 results, keeping the sets distinct."""
import json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'work/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
result=json.loads((ROOT/'outputs/mobilenet_cosface/comparison.json').read_text())
dest=ROOT/'outputs'
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
names=['Current CNN + ViT','MobileNet 224','MobileNet 384','MobileNet 384 + partial']
values=[result['baseline']['validation']['full']['closed_rank1']]+[r['validation']['full']['closed_rank1'] for r in result['candidates']]
axes[0].barh(names,np.array(values)*100,color=['#185d42','#97b0c7','#587da1','#8c709e'])
axes[0].set(xlim=(0,105),xlabel='Correct top match (%)',title='Development validation · full muzzle')
axes[0].axvline(90,color='#c27232',linestyle='--',linewidth=1)
for i,v in enumerate(values):axes[0].text(v*100+1,i,f'{v*100:.1f}%',va='center',fontsize=10)
axes[0].invert_yaxis()
x=np.arange(3)
for i,(key,color) in enumerate([('baseline','#185d42'),('candidate','#587da1')]):
 vals=[result['held_out_test'][key][v]['closed_rank1']*100 for v in ['full','left_half','right_half']]
 bars=axes[1].bar(x+(i-.5)*.36,vals,.36,label='CNN + ViT' if key=='baseline' else result['selection']['candidate'],color=color)
 axes[1].bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
axes[1].set(xticks=x,xticklabels=['Full','Left half','Right half'],ylim=(0,105),ylabel='Correct top match (%)',title='Held-out comparison · fixed thresholds')
axes[1].axhline(90,color='#c27232',linestyle='--',linewidth=1);axes[1].legend(loc='lower right',fontsize=9)
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.suptitle('MobileNetV3-L + CosFace: measured results',fontsize=15)
fig.savefig(dest/'mobilenet_cosface_accuracy.png',dpi=170)
