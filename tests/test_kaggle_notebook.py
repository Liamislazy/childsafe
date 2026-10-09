import ast
import json
from pathlib import Path
import pytest

@pytest.mark.parametrize('filename', ['childsafe_expanded_kaggle.ipynb', 'childsafe_kaggle.ipynb'])
def test_embedded_notebook_matches_current_source_and_parses(filename):
    root=Path(__file__).resolve().parents[1]
    nb=json.loads((root/'notebooks'/filename).read_text(encoding='utf-8'))
    assert nb['nbformat']==4
    ids=[cell['id'] for cell in nb['cells']]
    assert len(ids)==len(set(ids))
    for cell in nb['cells']:
        if cell['cell_type']!='code': continue
        source=''.join(cell['source'])
        if source.startswith('%%writefile '):
            command,body=source.split('\n',1)
            path=root/command.removeprefix('%%writefile ')
            expected=path.read_text(encoding='utf-8') or '# package\n'
            assert body==expected, f'Regenerate notebook after changing {path}'
            if path.suffix=='.py': ast.parse(body)
        else:
            ast.parse('\n'.join(line for line in source.splitlines() if not line.startswith('%')))
