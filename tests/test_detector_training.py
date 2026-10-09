import importlib.util
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def prepare_module():
    path=Path(__file__).resolve().parents[1]/'scripts/prepare_small_object_detector.py'
    spec=importlib.util.spec_from_file_location('prepare_detector',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_detector_labels_preserve_box_coordinates_and_separate_classes():
    module=prepare_module()
    assert module.CLASSES=={'Pen':'pen','Chopsticks':'chopsticks'}
    row=SimpleNamespace(XMin=.1,YMin=.2,XMax=.5,YMax=.8)
    assert module.yolo_box(row,1)=='1 0.30000000 0.50000000 0.40000000 0.60000000'
    row.XMax=1.1
    with pytest.raises(ValueError): module.yolo_box(row,1)


def test_detector_notebook_embeds_current_training_and_uses_heldout_test():
    root=Path(__file__).resolve().parents[1]
    notebook=json.loads((root/'notebooks/childsafe_detector_kaggle.ipynb').read_text(encoding='utf-8'))
    code=[''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type']=='code']
    expected=(root/'scripts/train_small_object_detector.py').read_text(encoding='utf-8').split("if __name__=='__main__':")[0]
    assert expected in code
    assert "split='test'" in expected
    for source in code:
        if not source.startswith('%'): ast.parse(source)
