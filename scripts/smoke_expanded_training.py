"""CPU plumbing check on real crops, NOT a deployment model or scientific benchmark."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from childsafe.config import load_config
from childsafe.classifier.train import main
from childsafe.vision.object_classifier import ObjectClassifier
from PIL import Image

def run():
    torch.set_num_threads(2)
    root=Path(__file__).resolve().parents[1]
    out=root/'outputs/verification/expanded_smoke'; out.mkdir(parents=True,exist_ok=True)
    source=root/'data/supplement'
    df=pd.read_csv(source/'manifest.csv')
    df=df[df.reviewed.astype(str).str.lower()=='true']
    parts=[]
    for category in ['pen','spoon']:
        part=df[df.category==category].head(14).copy()
        part['split_group']=['train']*8+['val']*2+['cal']*2+['test']*2
        part['path']=[str(source/p) for p in part.path]; parts.append(part)
    manifest=out/'manifest.csv'; pd.concat(parts).to_csv(manifest,index=False)
    cfg=load_config(); cfg['train'].update(img_size=64,num_workers=0,amp=False,patience=2)
    config=out/'config.yaml'; config.write_text(yaml.safe_dump(cfg),encoding='utf-8')
    main(['--arch','efficientnet_b0','--manifest',str(manifest),'--config',str(config),
          '--epochs','1','--batch-size','4','--no-pretrained','--balanced','--robust',
          '--preprocessing','letterbox','--out',str(out/'run')])
    classifier=ObjectClassifier(str(out/'run/best.pt'),'cpu')
    sample=np.array(Image.open(parts[0].path.iloc[0]).convert('RGB'))
    result=classifier(sample)
    assert classifier.classes==['pen','spoon']
    assert all(np.isfinite(result[k]) for k in ['p_unsafe_cal','obj_conf','p_unknown_cal'])
    assert json.loads((out/'run/metrics.json').read_text())['calibration_partition']=='cal'
    report={'purpose':'Plumbing test only; random initialization, one epoch, two classes',
            'training_completed':True,'checkpoint_reloaded':True,'prediction':result}
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__': run()
