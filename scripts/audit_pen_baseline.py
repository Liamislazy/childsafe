"""Baseline diagnostic on reviewed external pen crops; not an event benchmark."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from childsafe.vision.object_classifier import ObjectClassifier
from childsafe.config import load_config
from childsafe.alerts.processor import AlertProcessor

def main():
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1]
    source=root/'data/supplement'
    df=pd.read_csv(source/'manifest.csv')
    df=df[(df.category=='pen') & (df.reviewed.astype(str).str.lower()=='true')]
    model=ObjectClassifier(str(root/'outputs/runs/efficientnet_b0_group/best.pt'),'cpu')
    rows=[]
    for r in df.to_dict('records'):
        rgb=np.array(Image.open(source/r['path']).convert('RGB'))
        obj=model(rgb)
        old_conf=max(obj['p_unsafe_cal'],1-obj['p_unsafe_cal'])
        old_status='uncertain' if old_conf < .7 else ('unsafe' if obj['p_unsafe_cal']>=.5 else 'safe')
        processor=AlertProcessor(load_config())
        status=processor.step({'t':0,'hand_ok':1,'face_ok':1,'d_face_width':.1},obj)['obj_status']
        rows.append({'image_id':r['image_id'],**obj,'old_status':old_status,'new_status':status})
    result=pd.DataFrame(rows)
    out=root/'outputs/verification'; out.mkdir(exist_ok=True,parents=True)
    result.to_csv(out/'pen_baseline_predictions.csv',index=False)
    report={'n':len(rows),'domain':'External Open Images pen-only crops, not hand crops',
            'checkpoint':'Canonical 10-class efficientnet_b0_group; no pen class',
            'old_status_counts':result.old_status.value_counts().to_dict(),
            'new_status_counts':result.new_status.value_counts().to_dict(),
            'note':'Same old weights; only rejection logic differs. Not a retrained-model result or webcam event metric.'}
    (out/'pen_baseline_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
