import importlib

import pytest

ARMS = ('Q2_F', 'Q_A0', 'Q_P', 'M_C', 'M_N')


def api():
    return importlib.import_module('scripts.qp_mn_evaluation')


def rows():
    return [{'module': a, 'seed': 45, 'step': s, 'role': 'CAL', 'status': 'COMPLETE',
             'metrics': {'T2': .5, 'T1': .6}} for a in ARMS for s in (500, 1000, 1500)]


def test_cal_lock_requires_all_trajectories_and_breaks_ties_by_t1_then_earlier():
    data = rows()
    data[1]['metrics']['T1'] = .61
    data[2]['metrics']['T1'] = .61
    data[2]['metrics']['T2'] += .0000001
    result = api().build_cal_lock(data)
    assert result['selected_updates']['Q2_F'] == 1000
    assert result['selected_updates']['Q_P'] == 500
    with pytest.raises(ValueError):
        api().build_cal_lock(data[:-1])
    data[-1]['role'] = 'SEL'
    with pytest.raises(ValueError):
        api().build_cal_lock(data)


def test_sel_requires_global_lock_and_only_selected_or_fixed_endpoint():
    with pytest.raises(ValueError):
        api().validate_sel_request(None, module='B0', step=0, seed=45)
    lock = api().build_cal_lock(rows())
    api().validate_sel_request(lock, module='Q_P', step=500, seed=45)
    api().validate_sel_request(lock, module='Q_P', step=1500, seed=45)
    with pytest.raises(ValueError):
        api().validate_sel_request(lock, module='Q_P', step=1000, seed=45)
    with pytest.raises(ValueError):
        api().validate_sel_request(lock, module='Q_P', step=500, seed=46)


@pytest.mark.parametrize('d2,d1,expected', [(-.01,0,'NEGATIVE'),(0,0,'NO_GAIN'),
    (.001,-.003,'T2_ONLY_TRADEOFF'),(.001,None,'T2_ONLY_TRADEOFF'),
    (.001,0,'POSITIVE_SINGLE_SEED'),(.005,0,'TARGET_MAGNITUDE')])
def test_classification_uses_ap_units_and_t1_guard(d2,d1,expected):
    assert api().classify_gain(d2,d1) == expected
