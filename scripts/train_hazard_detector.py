"""Train the classes ACTUALLY annotated; keep person detection separate at runtime."""
import argparse
import hashlib
import json
from pathlib import Path

import yaml


def train(data, output, task='segment', epochs=60, size=960, batch=8, device='0',
          backbone=Path('models/yolo11n-seg.pt')):
    data,output,backbone = Path(data),Path(output),Path(backbone)
    if output.exists():
        raise FileExistsError('use a new output directory')
    if not backbone.is_file():
        raise FileNotFoundError('explicit local backbone required; never download implicitly')
    report = json.loads((data.parent/'report.json').read_text(encoding='utf-8'))
    if report['task'] != task:
        raise ValueError('dataset annotation task does not match training task')
    names = yaml.safe_load(data.read_text(encoding='utf-8'))['names']
    if list(names.values()) != report['classes']:
        raise ValueError('dataset class inventory differs from reviewed report')
    from ultralytics import YOLO
    architecture = 'yolo11n-seg.yaml' if task == 'segment' else 'yolo11n.yaml'
    model = YOLO(architecture).load(str(backbone))
    # Val controls model selection; test is used once after training, never for tuning.
    model.train(data=str(data.resolve()),epochs=epochs,imgsz=size,batch=batch,device=device,
                workers=2,project=str(output.parent),name=output.name,exist_ok=False,
                seed=42,deterministic=True,patience=15,amp=False,plots=True,
                cache=False,close_mosaic=10,save=True)
    best = Path(model.trainer.best); evaluated = YOLO(str(best))
    result = evaluated.val(data=str(data.resolve()),split='test',device=device,imgsz=size,
                           plots=True,project=str(output),name='heldout_test')
    metrics = result.seg if task == 'segment' else result.box
    payload = dict(task=task,classes=evaluated.names,checkpoint=best.name,
        checkpoint_sha256=hashlib.sha256(best.read_bytes()).hexdigest(),
        backbone_sha256=hashlib.sha256(backbone.read_bytes()).hexdigest(),
        dataset_manifest_sha256=hashlib.sha256((data.parent/'manifest.csv').read_bytes()).hexdigest(),
        test_metrics=result.results_dict,
        per_class={evaluated.names[int(cid)]:dict(precision=float(metrics.p[i]),recall=float(metrics.r[i]),
                   map50=float(metrics.ap50[i]),map50_95=float(metrics.ap[i]))
                   for i,cid in enumerate(metrics.ap_class_index)},
        limitation='Independent camera trials still required. Boxes only permit candidate proximity; no choking diagnosis.')
    (output/'evaluation.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(payload,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--output',type=Path,default=Path('outputs/hazard_detector'))
    parser.add_argument('--task',choices=['detect','segment'],default='segment')
    parser.add_argument('--epochs',type=int,default=60)
    parser.add_argument('--imgsz',type=int,default=960)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--device',default='0')
    parser.add_argument('--backbone',type=Path,default=Path('models/yolo11n-seg.pt'))
    args=parser.parse_args()
    train(args.data,args.output,args.task,args.epochs,args.imgsz,args.batch,args.device,args.backbone)
