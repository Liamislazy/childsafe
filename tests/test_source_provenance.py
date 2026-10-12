from childsafe.evaluation.sessions import code_hash


def test_git_line_endings_do_not_change_source_identity(tmp_path):
    path=tmp_path/'module.py'
    path.write_bytes(b'x = 1\n')
    original=code_hash(tmp_path)
    path.write_bytes(b'x = 1\r\n')
    assert code_hash(tmp_path)==original
    path.write_bytes(b'x = 2\r\n')
    assert code_hash(tmp_path)!=original
