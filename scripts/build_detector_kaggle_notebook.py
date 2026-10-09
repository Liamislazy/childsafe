"""Create a standalone Kaggle notebook for the Pen/Chopsticks detector."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build():
    cells=[]
    def cell(kind,text):
        item=dict(cell_type=kind,metadata={},source=text.splitlines(keepends=True),id=f'detector-{len(cells)}')
        if kind=='code': item.update(execution_count=None,outputs=[])
        cells.append(item)
    cell('markdown','''# ChildSafe — detector bút và đũa
Upload childsafe_pen_chopsticks_detection.zip thành Kaggle Dataset, thêm vào notebook.
Bật GPU và Internet. Dữ liệu là ảnh nguyên khung cùng bounding box, không phải crop phân loại.
Train/val/test giữ split chính thức của Open Images. Test chỉ dùng đánh giá cuối.
Đây là detector hộp: cần mask hoặc nhãn đầu vật bổ sung trước khi xác nhận khoảng cách đầu vật–miệng.
''')
    cell('code','%pip install -q ultralytics==8.3.253\n')
    cell('code','''from pathlib import Path
import os, shutil, yaml, torch
assert torch.cuda.is_available(), 'Enable GPU in Kaggle notebook settings'
matches=list(Path('/kaggle/input').rglob('manifest.csv'))
matches=[p.parent for p in matches if (p.parent/'report.json').is_file() and (p.parent/'images/train').is_dir()]
assert len(matches)==1, f'Expected one detector dataset, found {matches}'
source=matches[0]
work=Path('/kaggle/working/childsafe_detector'); work.mkdir(exist_ok=True)
os.chdir(work)
(work/'models').mkdir(exist_ok=True)
weights=list(Path('/kaggle/input').rglob('yolo11n-seg.pt'))
assert len(weights)==1, 'Dataset ZIP must contain models/yolo11n-seg.pt'
shutil.copy2(weights[0],work/'models/yolo11n-seg.pt')
cfg={'path':str(source),'train':'images/train','val':'images/val','test':'images/test','names':{0:'pen',1:'chopsticks'}}
Path('dataset.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
print((source/'report.json').read_text())
''')
    cell('code',(ROOT/'scripts/train_small_object_detector.py').read_text(encoding='utf-8').split("if __name__=='__main__':")[0])
    cell('code',"train(Path('dataset.yaml'), Path('outputs/pen_chopsticks_detector_gpu'), epochs=50, size=640, batch=16, device='0')\n")
    cell('code','''import zipfile
with zipfile.ZipFile('/kaggle/working/pen_chopsticks_detector_results.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for p in Path('outputs').rglob('*'):
        if p.is_file(): archive.write(p,p.as_posix())
print('Download pen_chopsticks_detector_results.zip from Kaggle Output')
''')
    notebook=dict(nbformat=4,nbformat_minor=5,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},cells=cells)
    target=ROOT/'notebooks/childsafe_detector_kaggle.ipynb'
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    print(target)


if __name__=='__main__': build()
