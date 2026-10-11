from copy import deepcopy

import pytest

from scripts.native_long_mechanism_replay import validate_plan


def test_replay_preserves_original_inputs_weights_and_full_repeat_coverage():
    original = {'records': ['immutable'], 'data_files': ['immutable'], 'population_sha256': 'p',
                'metric_spec_sha256': 'm', 'cpu_preflight_sha256': 'c', 'code': {'model.py': 'code'},
                'jobs': [{'name': f'{i}-{mode}', 'arm': 'E3', 'step': 2970,
                          'record_index': i, 'mode': mode, 'bundle_sha256': 'weights'}
                         for i in (21, 25) for mode in ('normal', 'off')]}
    plan = deepcopy(original)
    plan.update(total_ceiling_gpu_hours=.3, jobs=[{**j, 'name': f"r{r}-{j['name']}",
                'original_job': j['name'], 'repeat': r} for j in original['jobs'] for r in (1, 2)])
    validate_plan(plan, original)
    changed = deepcopy(plan)
    changed['jobs'][0]['bundle_sha256'] = 'new-weights'
    with pytest.raises(ValueError, match='checkpoint'):
        validate_plan(changed, original)
    changed = deepcopy(plan)
    changed['jobs'][-1] = changed['jobs'][0]
    with pytest.raises(ValueError, match='coverage'):
        validate_plan(changed, original)
    changed = deepcopy(plan)
    changed['code']['model.py'] = 'edited'
    with pytest.raises(ValueError, match='numerical source'):
        validate_plan(changed, original)
