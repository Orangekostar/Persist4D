"""Official pooled AP with physical-reference pairing and preserved NA draws."""

import math
from pathlib import Path
import statistics

from scripts.native_long_budget import write_json_atomic
from scripts.ogr_assets import ARTIFACTS, digest, read_json, sha256


def conditional_interval(values):
    """The fixed 950/1000 threshold is applied without redrawing invalid draws."""
    import numpy as np
    valid = [float(value) for value in values if value is not None and math.isfinite(value)]
    return dict(draws=len(values), valid_draws=len(valid), NA_draws=len(values)-len(valid),
        NA_fraction=(len(values)-len(valid))/len(values) if values else None,
        conditional_valid_draw_interval=np.percentile(valid, [2.5, 97.5]).tolist() if len(valid) >= 950 else None,
        interval_status='CONDITIONAL_ON_VALID_DRAWS' if len(valid) >= 950 else 'NA_FEWER_THAN_950_VALID_DRAWS',
        redraw_invalid=False)


def signed_difference(left, right):
    return None if left is None or right is None else left-right


def paired_seed_mean(values):
    return None if any(value is None for value in values) else statistics.mean(values)


def classify_evidence(evidence):
    """A conservative internal evidence label; absent evidence is never success."""
    if not evidence.get('core_complete'):
        return dict(PRIMARY='INCONCLUSIVE', mechanism='NOT_TESTED', target_supervision='NOT_TESTED',
                    association='NOT_TESTED', reason=evidence.get('reason', 'formal comparison incomplete'))
    supported = (evidence.get('both_seeds_positive_SEL_PB_long_current') is True
        and evidence.get('all_seed_mean_interval_lower_positive') is True
        and evidence.get('OBS_current_above_ID_QFB') is True
        and evidence.get('OBS_shared_target_iou_above_ID_QFB') is True
        and evidence.get('coverage_unchanged') is True
        and evidence.get('numeric_confound') is False)
    labels = []
    if supported:
        labels.append('MECHANISM_SUPPORTED_INTERNAL')
    if evidence.get('TARGET_improved') and not evidence.get('OBS_extra_evidence'):
        labels.append('TARGET_SUPERVISION_ONLY')
    if evidence.get('ID_identity_improved') and not evidence.get('OBS_mask_current_improved'):
        labels.append('ASSOCIATION_ONLY')
    if evidence.get('point_estimate_positive') and not supported:
        labels.append('PROMISING_INCONCLUSIVE')
    if evidence.get('cannot_exceed_baseline_controls') or evidence.get('clear_current_or_recovery_cost'):
        labels.append('SCREEN_NO_GO')
    return dict(PRIMARY=labels[0] if len(labels) == 1 else 'INCONCLUSIVE', dimensions=labels,
                venue_readiness='NOT_INFERRED_FROM_INTERNAL_LABEL')


def _alignment_check(banks, assets):
    """Exactly two actual nonempty pools, once for this evidence population."""
    from scripts.evaluate_revisit_prompt import pool_official_states
    from scripts.revisit_prompt_cpu_statistics import accelerated_precision_recall

    marker = ARTIFACTS / 'CPU_POOL_ALIGNMENT.json'
    identity = digest({str(key): state['sha256'] for key, state in banks.items()})
    if marker.is_file():
        previous = read_json(marker)
        if previous['population_sha256'] != identity:
            raise ValueError('CPU alignment evidence refers to a different state population')
        return previous
    keys = sorted(banks)
    if len(keys) < 2:
        raise ValueError('two real official states are required for CPU alignment')
    comparisons = []
    for chosen in (keys[:1], keys[:2]):
        states = [banks[key] for key in chosen]
        if len({state['mode'] for state in states}) != 1:
            states = [states[0]]
        original = pool_official_states(states, dataset_spec=assets['taxonomy_path'])
        with accelerated_precision_recall():
            accelerated = pool_official_states(states, dataset_spec=assets['taxonomy_path'])
        if original != accelerated:
            raise ValueError('accelerated CPU official pooling differs from the original evaluator')
        comparisons.append(dict(state_sha256=[state['sha256'] for state in states], original=original,
                                accelerated=accelerated, exact=True))
    result = dict(status='PASS', real_pools=2, population_sha256=identity, comparisons=comparisons)
    write_json_atomic(marker, result)
    return result


def bootstrap_role(assets, *, role, endpoints, control):
    """Use one set of 1000 reference multisets for every method and seed."""
    import numpy as np
    from scripts.evaluate_revisit_prompt import pool_official_states
    from scripts.revisit_prompt_cpu_statistics import accelerated_precision_recall
    from scripts.ogr_selection import evaluation_folder, verified_evaluation

    records = read_json(ARTIFACTS / 'DATA_MANIFEST.json')['records'][role]
    references = sorted({record['reference_id'] for record in records})
    banks, state_bindings = {}, []
    for (method, seed), step in endpoints.items():
        result = verified_evaluation(assets, method, seed, step, role)
        folder = evaluation_folder(assets, method, seed, step, role)
        for row in result['references']:
            path = folder / row['evidence']
            state = read_json(path)
            banks[method, seed, row['kind'], row['horizon'], row['reference_id']] = state
            state_bindings.append(dict(method=method, seed=seed, step=step, role=role,
                kind=row['kind'], horizon=row['horizon'], reference_id=row['reference_id'], sha256=row['sha256']))
    _alignment_check_for_role = ARTIFACTS / 'CPU_POOL_ALIGNMENT.json'
    if not _alignment_check_for_role.exists():
        _alignment_check(banks, assets)
    # Immutable, already verified bank + multiset counts is a sufficient cache
    # key. Hashing large raw states repeatedly inside each draw is unnecessary.
    cache = {}
    def score(method, seed, kind, counts):
        key = method, seed, kind, tuple(counts)
        if key in cache:
            return cache[key]
        values = []
        for horizon in (3, 4, 5):
            states = [banks[method, seed, kind, horizon, reference]
                for reference, count in zip(references, counts, strict=True)
                if (method, seed, kind, horizon, reference) in banks for _ in range(int(count))]
            if not states:
                values.append(None)
                continue
            metrics = pool_official_states(states, dataset_spec=assets['taxonomy_path'])
            values.append(metrics['online_t-mAP' if kind == 'long' else 'raw_local_AP'])
        value = None if any(item is None for item in values) else statistics.mean(values)
        cache[key] = value
        return value
    generator = np.random.default_rng(9101)
    draws = generator.integers(0, len(references), size=(1000, len(references)))
    multisets = [np.bincount(draw, minlength=len(references)).tolist() for draw in draws]
    comparisons = {(method, kind): [] for method in ('R-TARGET', 'R-ID', 'R-QFB', 'TWO_SEED_MEAN_CONTROL_STAR')
                   for kind in ('long', 'current')}
    point_estimates = {}
    with accelerated_precision_recall():
        for counts in [[1]*len(references), *multisets]:
            values = {}
            for method in ('R-TARGET', 'R-ID', 'R-QFB'):
                for kind in ('long', 'current'):
                    values[method, kind] = signed_difference(score('R-OBS', 246, kind, counts), score(method, 246, kind, counts))
            for kind in ('long', 'current'):
                values['TWO_SEED_MEAN_CONTROL_STAR', kind] = paired_seed_mean([
                    signed_difference(score('R-OBS', seed, kind, counts), score(control, seed, kind, counts))
                    for seed in (246, 247)])
            if not point_estimates:
                point_estimates = {f'{method}:{kind}': value for (method, kind), value in values.items()}
            else:
                for key, value in values.items():
                    comparisons[key].append(value)
    return dict(role=role, bootstrap_seed=9101, samples=1000, unit='physical_reference',
        independent_reference_count=len(references), small_reference_warning=len(references) < 10,
        shared_reference_multisets=True, reference_ids=references, multiset_counts=multisets,
        draws_sha256=digest(multisets), state_bindings=state_bindings,
        comparisons={f'{method}:{kind}': dict(point_difference=point_estimates[f'{method}:{kind}'], **conditional_interval(values))
                     for (method, kind), values in comparisons.items()},
        multi_seed_rule='difference within each seed, then mean; predictions from seeds are never concatenated',
        pooled_cache_entries=len(cache))


def compute_statistics(assets):
    plan = read_json(ARTIFACTS / 'RUN_PLAN.json')
    if plan['status'] != 'LOCKED':
        result = dict(status='NOT_TESTED_BUDGET_BLOCKED', reason='NO_FEASIBLE_COMMON_N',
            samples_executed=0, official_pools_available=0, comparisons={},
            science=classify_evidence(dict(core_complete=False, reason='formal training and evaluation were not started')))
        write_json_atomic(ARTIFACTS / 'BOOTSTRAP_RESULTS.json', result)
        return result
    first = read_json(ARTIFACTS / 'CAL_SELECTION_LOCK.json')
    second = read_json(ARTIFACTS / 'REPLICATION_LOCK.json')
    control = first['control']['point']['variant']
    endpoints = {(method, seed): row['point']['step'] for seed, lock in ((246, first), (247, second))
                 for method, row in lock['endpoints'].items()}
    result = dict(status='COMPLETE', roles={role: bootstrap_role(assets, role=role, endpoints=endpoints, control=control)
                                           for role in ('SEL', 'PB')},
        science=classify_evidence(dict(core_complete=True)))
    write_json_atomic(ARTIFACTS / 'BOOTSTRAP_RESULTS.json', result)
    return result
