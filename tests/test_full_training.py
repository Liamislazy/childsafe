import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_full_training', ROOT/'scripts/prepare_full_training.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def full_inputs(tmp_path):
    raw, oi = tmp_path/'childsun', tmp_path/'openimages'
    raw.mkdir(); oi.mkdir()
    rows, downloads = [], []
    for source, folder in module.SOURCE_FOLDERS.items():
        for category in ('knife', 'other'):
            for role in ('train', 'val', 'val', 'test'):
                i = len(rows)
                name = f'{category}/{i:016x}_crop.jpg'
                root = raw if source == 'childsun' else oi/folder
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                rows.append(dict(source=source, path=name, category=category,
                                 label='unsafe' if category == 'knife' else 'safe',
                                 group=i, source_id=f'{source}_{i}', split_group=role))
                if folder:
                    downloads.append(dict(split=folder, path=f'{folder}/{name}',
                                          status='ok', image_id=f'{i:016x}'))
    test = oi/'test/knife/external.jpg'; test.parent.mkdir(parents=True); test.touch()
    downloads.append(dict(split='test', path='test/knife/external.jpg', status='ok', image_id='external'))
    pd.DataFrame(downloads).to_csv(oi/'manifest.csv', index=False)
    return pd.DataFrame(rows), raw, oi


def test_full_pool_preserves_test_and_train_and_reserves_whole_calibration_groups(full_inputs):
    base, raw, oi = full_inputs
    prepared = module.prepare_manifest(base, raw, oi)
    assert len(prepared) == len(base)
    for role in ('train', 'test'):
        assert set(prepared.loc[prepared.split_group == role, 'group']) == set(base.loc[base.split_group == role, 'group'])
    assert prepared.groupby('group').split_group.nunique().max() == 1
    for role in ('train', 'val', 'cal', 'test'):
        assert set(prepared.loc[prepared.split_group == role, 'category']) == {'knife', 'other'}
    assert all(Path(p).is_absolute() and Path(p).is_file() for p in prepared.path)
    assert not any('external.jpg' in p for p in prepared.path)
    pd.testing.assert_frame_equal(prepared, module.prepare_manifest(base, raw, oi))


def test_missing_training_crop_fails_instead_of_training_subset(full_inputs):
    base, raw, oi = full_inputs
    (oi/'train'/base.loc[base.source == 'openimages_train', 'path'].iloc[0]).unlink()
    with pytest.raises(ValueError, match='missing'):
        module.prepare_manifest(base, raw, oi)


def test_external_test_source_overlap_is_rejected(full_inputs):
    base, raw, oi = full_inputs
    manifest = pd.read_csv(oi/'manifest.csv')
    manifest.loc[manifest['split'] == 'test', 'image_id'] = manifest.loc[manifest['split'] == 'train', 'image_id'].iloc[0]
    manifest.to_csv(oi/'manifest.csv', index=False)
    with pytest.raises(ValueError, match='overlap'):
        module.prepare_manifest(base, raw, oi)


def test_pinned_full_split_has_expected_complete_sources():
    base = pd.read_csv(ROOT/'data/splits/splits_with_other_train.csv')
    assert base.groupby('source').size().to_dict() == {'childsun': 5353, 'openimages': 1973, 'openimages_train': 20557}
    assert base.groupby('group').split_group.nunique().max() == 1
    assert base.category.nunique() == 11


def test_preparation_cli_writes_manifest_and_provenance(full_inputs, tmp_path):
    base, raw, oi = full_inputs
    pinned = tmp_path/'pinned.csv'; base.to_csv(pinned, index=False)
    out = tmp_path/'outputs/full.csv'
    module.main(['--base', str(pinned), '--raw-root', str(raw),
                 '--openimages-root', str(oi), '--out', str(out)])
    prepared = pd.read_csv(out)
    report = json.loads(out.with_suffix('.report.json').read_text())
    assert len(prepared) == report['images'] == len(base)
    assert report['classes'] == ['knife', 'other']
    assert len(report['base_sha256']) == 64
    assert set(prepared.split_group) == {'train', 'val', 'cal', 'test'}
