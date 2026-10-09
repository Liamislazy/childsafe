"""Train a separate small-object detector; boxes are not segmentation masks."""
import argparse
import json
from pathlib import Path


def train(data, output, epochs, size, batch, device):
    if output.exists():
        raise FileExistsError('training output exists; choose a new --output directory')
    from ultralytics import YOLO
    # Transfer the existing COCO backbone into a detection architecture. The
    # two-class detection head is trained afresh; no weight download is needed.
    model=YOLO('yolo11n.yaml').load('models/yolo11n-seg.pt')
    model.train(data=str(data),epochs=epochs,imgsz=size,batch=batch,device=device,
                workers=0,project=str(output.parent),name=output.name,exist_ok=False,
                seed=42,deterministic=True,patience=10,amp=False,plots=False,
                cache=False,close_mosaic=3,save=True)
    best=Path(model.trainer.best)
    evaluated=YOLO(str(best))
    result=evaluated.val(data=str(data),split='test',device=device,imgsz=size,plots=False,
                         project=str(output),name='heldout_test')
    report={'checkpoint':str(best.resolve()),'classes':evaluated.names,'epochs_requested':epochs,
            'image_size':size,'test_metrics':result.results_dict,
            'limitation':'Bounding-box baseline. Does not localize object tips or confirm mask proximity.'}
    report['per_class']={evaluated.names[int(class_id)]:dict(precision=float(result.box.p[i]),
                        recall=float(result.box.r[i]),map50=float(result.box.ap50[i]),
                        map50_95=float(result.box.ap[i]))
                        for i,class_id in enumerate(result.box.ap_class_index)}
    output.mkdir(parents=True,exist_ok=True)
    (output/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('data/detector/pen_chopsticks/dataset.yaml'))
    parser.add_argument('--output',type=Path,default=Path('outputs/pen_chopsticks_detector_cpu'))
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--imgsz',type=int,default=384)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    train(args.data,args.output,args.epochs,args.imgsz,args.batch,args.device)
