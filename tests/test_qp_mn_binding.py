import importlib
import json

import pytest


def api():
    return importlib.import_module('scripts.qp_mn_binding')


def test_ledger_dedup_excludes_subtotal_but_counts_failed_attempts(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    a.write_text('\n'.join(json.dumps(r) for r in [
        {'event_id': 'one', 'gpu_hours': 2}, {'event_id': 'two', 'gpu_hours': .5, 'status': 'FAILED'}]))
    b.write_text('\n'.join(json.dumps(r) for r in [
        {'event_id': 'subtotal', 'gpu_hours': 2.5}, {'event_id': 'two', 'gpu_hours': .5, 'status': 'FAILED'},
        {'event_id': 'three', 'gpu_hours': 1}]))
    result = api().ledger_total([a, b], subtotal_ids={'subtotal'})
    assert result['gpu_hours'] == 3.5 and result['unique_events'] == 3
    b.write_text(json.dumps({'event_id': 'one', 'gpu_hours': 3}))
    with pytest.raises(ValueError):
        api().ledger_total([a, b], subtotal_ids=set())


def test_input_validation_uses_content_not_complete_flag(tmp_path):
    p = tmp_path / 'pred'
    p.write_bytes(b'changed')
    index = {'status': 'COMPLETE', 'cache': str(tmp_path), 'entries': [
        {'input_id': 'x', 'prediction': {'file': 'pred', 'sha256': 'wrong'}}]}
    with pytest.raises(ValueError):
        api().verify_index_files(index, fields=('prediction',))
