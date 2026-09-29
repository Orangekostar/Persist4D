import importlib
from pathlib import Path


def test_stage_receipt_cannot_reuse_changed_output_or_dependency(tmp_path):
    api = importlib.import_module('scripts.qp_mn_campaign')
    output = tmp_path / 'result.json'
    output.write_text('{"value": 1}\n')
    receipt = {'identity': {'label': 'v2'}, 'outputs': {str(output): api.file_digest(output)}}
    assert api.receipt_valid(receipt, {'label': 'v2'})
    assert not api.receipt_valid(receipt, {'label': 'v3'})
    output.write_text('{"value": 2}\n')
    assert not api.receipt_valid(receipt, {'label': 'v2'})
    Path(output).unlink()
    assert not api.receipt_valid(receipt, {'label': 'v2'})
