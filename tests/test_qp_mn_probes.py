import importlib

import pytest
import torch


def test_fit_draws_are_fixed_positive_positions_and_never_resampled():
    api = importlib.import_module('scripts.qp_mn_probes')
    rows = [{'input_id':'r','source_query_id':i,'source_class_id':0,'retained_index':i,'H':1,
             'reference_id':'ref'} for i in reversed(range(8))]
    records = {'r':{'quality':{'geometry_valid':torch.ones(8,dtype=torch.bool)},
                    'assignment':torch.arange(8), 'weights':torch.ones(3)}}
    draws = api.make_fit_draws(rows,records)
    assert [r['candidates'] for r in draws] == [[i] for i in range(8)]
    assert [r['segment_seed'] for r in draws] == list(range(145,153))
    records['r']['quality']['geometry_valid'][3] = False
    with pytest.raises(ValueError):
        api.make_fit_draws(rows,records)
