import importlib


def test_replication_pairs_new_method_with_control_at_locked_steps_only():
    api = importlib.import_module('scripts.qp_mn_results')
    arms = ['Q2_F','Q_A0','Q_P','M_C','M_N']
    lock = {'selected_updates': dict(zip(arms,[1500,1000,500,500,1500]))}
    rows = [{'module': a, 'seed': 45, 'role': 'SEL', 'step': lock['selected_updates'][a],
             'status': 'COMPLETE', 'metrics': {'T1': .8, 'T2': .69}, 'by_reference': []} for a in arms]
    rows[2]['metrics']['T2'] = .702
    rows[4]['metrics']['T2'] = .703
    rows[4]['metrics']['T1'] = .79  # T1 protection fails; no M-N replication.
    plan = api.build_replication_plan({'baseline': {'metrics': {'T1': .8, 'T2': .7}}, 'rows': rows}, lock)
    assert plan['steps'] == {'Q_A0': [500,1000], 'Q_P': [500]}
    assert plan['candidates'] == ['Q_P']
    assert plan['schedule_horizon'] == 1500
