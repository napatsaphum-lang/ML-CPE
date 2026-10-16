import csv
from pathlib import Path
from ultralytics import YOLO
from .data_loader import ROOT

def evaluate_one(weights, yaml_path):
    result = YOLO(str(weights)).val(data=str(yaml_path), split='test', plots=True, project=str(ROOT/'outputs'),name=f'eval_{weights.parent.parent.name}',exist_ok=True,device='cpu',workers=0)
    return {'model': weights.parent.parent.name, 'precision':round(float(result.box.mp),4), 'recall':round(float(result.box.mr),4), 'mAP50':round(float(result.box.map50),4), 'mAP50-95':round(float(result.box.map),4)}

def save_table(rows):
    p = ROOT/'outputs'/'comparison.csv'
    p.parent.mkdir(exist_ok=True)
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['model','precision','recall','mAP50','mAP50-95'])
        writer.writeheader(); writer.writerows(rows)
    print('Comparison saved:',p)
    try:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,5))
        x=range(len(rows)); width=.2
        for j,k in enumerate(['precision','recall','mAP50','mAP50-95']):
            ax.bar([i+(j-1.5)*width for i in x],[r[k] for r in rows],width,label=k)
        ax.set_xticks(list(x),[r['model'] for r in rows]);ax.set_ylim(0,1);ax.set_ylabel('Score');ax.set_title('LAB10 - YOLO Model Comparison');ax.legend()
        fig.tight_layout();fig.savefig(ROOT/'outputs'/'comparison.png',dpi=150);plt.close(fig)
    except ImportError:
        print('Install matplotlib to create comparison.png')
