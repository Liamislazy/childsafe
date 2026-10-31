"""Generate the standalone multiclass detector notebook, without execution/results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells=[]
    def add(kind,source):
        cell=dict(cell_type=kind,metadata={},id=f'hazard-{len(cells)}',source=source.splitlines(keepends=True))
        if kind=='code':
            cell.update(execution_count=None,outputs=[])
        cells.append(cell)
    add('markdown','''# ChildSafe — detector vật nguy cơ mở rộng
Upload ZIP từ package_hazard_dataset.py thành một Kaggle Dataset riêng, rồi thêm vào notebook.
Chỉ các lớp có nhãn thật ở cả train/val/test mới được train. Không tự sinh nhãn từ model.
Bật GPU. Internet chỉ cần để cài dependency. Backbone phải có trong gói local.
Ưu tiên segmentation để đo vùng vật; bounding box chỉ dùng cảnh giác nghi ngờ.
Kết quả ảnh tĩnh không chứng minh hiệu quả camera, kích thước vật lý hay nhận diện nghẹn.
''')
    add('code','%pip install -q ultralytics==8.3.253\n')
    add('code','''from pathlib import Path
import json, yaml, torch
assert torch.cuda.is_available(), 'Enable GPU in Kaggle settings'
candidates=[p for p in Path('/kaggle/input').rglob('dataset.yaml') if (p.parent/'report.json').is_file()]
assert len(candidates)==1, f'Expected one reviewed object dataset, found {candidates}'
source=candidates[0].parent
report=json.loads((source/'report.json').read_text())
backbones=list(Path('/kaggle/input').rglob('yolo11n-seg.pt'))
assert len(backbones)==1, 'Package one explicit local backbone with the dataset'
work=Path('/kaggle/working/hazard_data'); work.mkdir(exist_ok=False)
cfg=yaml.safe_load((source/'dataset.yaml').read_text()); cfg['path']=str(source)
(work/'dataset.yaml').write_text(yaml.safe_dump(cfg))
import shutil
for name in ('manifest.csv','report.json'): shutil.copy2(source/name, work/name)
print(report)
''')
    add('code',(ROOT/'scripts/train_hazard_detector.py').read_text(encoding='utf-8').split("if __name__ == '__main__':")[0])
    add('code', '''train(work/'dataset.yaml', Path('/kaggle/working/outputs/hazard_detector'),
      task=report['task'], epochs=60, size=960, batch=8, device='0', backbone=backbones[0])
''')
    add('code','''import zipfile
with zipfile.ZipFile('/kaggle/working/childsafe_hazard_detector_results.zip','x',zipfile.ZIP_DEFLATED) as archive:
    root=Path('/kaggle/working/outputs/hazard_detector')
    for path in root.rglob('*'):
        if path.is_file(): archive.write(path,'hazard_detector/'+path.relative_to(root).as_posix())
print('Download childsafe_hazard_detector_results.zip; evaluate on independent camera sessions before deployment.')
''')
    notebook=dict(nbformat=4,nbformat_minor=5,
                  metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},cells=cells)
    target=ROOT/'notebooks/childsafe_hazard_detector_kaggle.ipynb'
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    return target


if __name__ == '__main__':
    print(build())
