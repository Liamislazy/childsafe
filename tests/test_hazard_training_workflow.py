"""Dataset integrity and standalone Kaggle notebook; no training/network/hardware."""
import ast
import csv
import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    obj=importlib.util.module_from_spec(spec); spec.loader.exec_module(obj)
    return obj


def dataset(tmp_path, task='segment'):
    rows=[]
    for i,split in enumerate(('train','val','test')):
        image=tmp_path/f'{split}.png'
        Image.new('RGB',(40,30),(50+i,80,90)).save(image)
        image.with_suffix('.txt').write_text('3 .1 .1 .8 .1 .8 .8 .1 .8\n' if task=='segment' else '3 .5 .5 .4 .4\n')
        rows.append(dict(image=image.name,split=split,group_id=f'adult{i}',reviewed=True,
                         annotation_complete=True,source='consented test fixture',license='synthetic',author='',url=''))
    manifest=tmp_path/'annotations.csv'
    def write():
        with manifest.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write()
    return manifest,rows,write


@pytest.mark.parametrize('task',['detect','segment'])
def test_dataset_preserves_independent_groups_and_only_trains_real_classes(tmp_path,task):
    manifest,_,_=dataset(tmp_path,task)
    out=tmp_path/'prepared'
    report=module('prepare_hazard_dataset').prepare(manifest,out,task)
    assert report['classes']==['pen'] and 'coin' in report['not_yet_annotated']
    assert yaml.safe_load((out/'dataset.yaml').read_text())['names']=={0:'pen'}
    assert next((out/'labels/train').glob('*.txt')).read_text().startswith('0 ')
    with pytest.raises(FileExistsError): module('prepare_hazard_dataset').prepare(manifest,out,task)
    backbone=tmp_path/'backbone.pt'; backbone.write_bytes(b'fixture')
    assert module('package_hazard_dataset').package(out,tmp_path/'dataset.zip',backbone)>0


@pytest.mark.parametrize('failure',['group','duplicate','review','box_as_mask','missing_label','missing_class','bad_box'])
def test_bad_data_fails_before_creating_a_training_dataset(tmp_path,failure):
    manifest,rows,write=dataset(tmp_path)
    first=tmp_path/rows[0]['image']; second=tmp_path/rows[1]['image']
    if failure=='group': rows[1]['group_id']=rows[0]['group_id']
    elif failure=='duplicate': second.write_bytes(first.read_bytes())
    elif failure=='review': rows[0]['annotation_complete']=False
    elif failure=='box_as_mask': first.with_suffix('.txt').write_text('3 .5 .5 .4 .4')
    elif failure=='missing_label': first.with_suffix('.txt').unlink()
    elif failure=='missing_class': first.with_suffix('.txt').write_text('')
    elif failure=='bad_box': first.with_suffix('.txt').write_text('3 1.1 .5 .4 .4')
    write(); output=tmp_path/'prepared'
    with pytest.raises(ValueError): module('prepare_hazard_dataset').prepare(manifest,output)
    assert not output.exists()


def test_notebook_embeds_current_trainer_and_keeps_test_for_final_evaluation():
    book=json.loads((ROOT/'notebooks/childsafe_hazard_detector_kaggle.ipynb').read_text(encoding='utf-8'))
    sources=[''.join(c['source']) for c in book['cells'] if c['cell_type']=='code']
    trainer=(ROOT/'scripts/train_hazard_detector.py').read_text(encoding='utf-8').split("if __name__ == '__main__':")[0]
    assert trainer in sources and "split='test'" in trainer
    for source in sources:
        if not source.startswith('%'): ast.parse(source)
    assert all(not cell.get('outputs') for cell in book['cells'])
