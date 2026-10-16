"""CPU-only same-class cooccurrence diagnostics from locked predictions and GT.

Conditional current-mask recall is descriptive; it is neither AP nor a causal
test of interference. Unknown/ambiguous identities leave only this diagnostic.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

VARIANTS = ('M0', 'M3', 'M4')
COUNT_FIELDS = ('current_gt', 'excluded_identity_gt', 'excluded_class_gt',
                'eligible_gt', 'cooccurring_gt', 'same_class_pairs')
MATCHING = 'class-aware one-to-one Hungarian IoU>=0.5; all official candidates including ephemeral'


def cooccurrence_counts(ids, labels, matched_by_variant, *, ambiguities, valid_class_ids):
    """Count distinct reliable entities and unordered same-class pairs once."""
    from models.task_memory_supervision import ambiguous_identity_ids

    if (len(ids) != len(labels) or any(type(i) is not int for i in ids)
            or any(type(label) is not int for label in labels) or len(set(ids)) != len(ids)
            or set(matched_by_variant) != set(VARIANTS)):
        raise ValueError('truth/match population is malformed or misaligned')
    population = set(ids)
    matched = {}
    for variant, values in matched_by_variant.items():
        if (any(type(i) is not int for i in values) or len(set(values)) != len(values)
                or not set(values) <= population):
            raise ValueError('matched population differs from current truth')
        matched[variant] = set(values)
    excluded = ambiguous_identity_ids(ambiguities)
    classes, valid = defaultdict(set), set(valid_class_ids)
    excluded_identity = excluded_class = 0
    for identity, label in zip(ids, labels):
        if identity < 0 or identity in excluded:
            excluded_identity += 1
        elif label not in valid:
            excluded_class += 1
        else:
            classes[label].add(identity)
    same = set().union(*(values for values in classes.values() if len(values) >= 2))
    counts = {variant: len(values & same) for variant, values in matched.items()}
    paired = {}
    for control in ('M0', 'M3'):
        recovered = len((matched['M4'] - matched[control]) & same)
        lost = len((matched[control] - matched['M4']) & same)
        paired['M4_vs_' + control] = dict(recovered=recovered, lost=lost, net=recovered-lost)
    return dict(current_gt=len(ids), excluded_identity_gt=excluded_identity,
                excluded_class_gt=excluded_class, eligible_gt=sum(map(len, classes.values())),
                cooccurring_gt=len(same), same_class_pairs=sum(
                    len(values)*(len(values)-1)//2 for values in classes.values()),
                matched_by_variant=counts,
                recall_by_variant={v: counts[v]/len(same) if same else None for v in VARIANTS},
                paired=paired)


def _summary(rows):
    result = {field: sum(row[field] for row in rows) for field in COUNT_FIELDS}
    result.update(input_events=len(rows), independent_references=len({row['reference_id'] for row in rows}),
                  inputs_with_same_class_entities=sum(row['cooccurring_gt'] > 0 for row in rows))
    result['matched_by_variant'] = {v: sum(row['matched_by_variant'][v] for row in rows) for v in VARIANTS}
    result['recall_by_variant'] = {
        v: count/result['cooccurring_gt'] if result['cooccurring_gt'] else None
        for v, count in result['matched_by_variant'].items()}
    result['paired'] = {pair: {field: sum(row['paired'][pair][field] for row in rows)
                              for field in ('recovered', 'lost', 'net')}
                        for pair in ('M4_vs_M0', 'M4_vs_M3')}
    return result


def _saved_diagnostic(row):
    diagnostics = row['current_mask_diagnostics']
    if len(diagnostics) != row['horizon']:
        raise ValueError('saved stage population differs')
    diagnostic = diagnostics[-1]
    matched, unmatched = diagnostic['matched_gt_ids'], diagnostic['unmatched_gt_ids']
    ids = matched + unmatched
    if (diagnostic['stage'] != row['horizon']-1 or diagnostic['matching'] != MATCHING
            or any(type(i) is not int for i in ids) or len(set(ids)) != len(ids)
            or len(ids) != diagnostic['gt_count']):
        raise ValueError('saved current GT population or matching rule differs')
    return set(ids), matched


def run_analysis(assets):
    """Read actual final observations once; compare all three saved processes."""
    if os.environ.get('MPR_CHARGED_EVENT_ID'):
        raise ValueError('offline diagnostic cannot run inside a charged GPU child')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import fcntl
    import torch
    import yaml
    from datasets.revisit_prompt_episode import RevisitEpisodeLoader, RevisitEpisodeSpec, causal_prefix_truth
    from models.task_memory_supervision import ambiguous_identity_ids
    from scripts.native_long_budget import write_json_atomic
    from scripts.revisit_prompt_assets import ARTIFACTS, PROJECT, parent_config, read_json, sha256

    if torch.cuda.is_initialized():
        raise ValueError('offline diagnostic requires an uninitialized CUDA runtime')
    torch.set_num_threads(2)
    output = ARTIFACTS / 'SAME_CLASS_DIAGNOSTICS.json'
    if output.exists():
        raise FileExistsError('preserve the existing same-class diagnostic artifact')
    manifest = read_json(ARTIFACTS / 'DATA_MANIFEST.json')
    lock = read_json(ARTIFACTS / 'RUN_LOCK.json')
    selection = read_json(ARTIFACTS / 'CAL_SELECTION_LOCK.json')
    policy = read_json(ARTIFACTS / 'EVALUATION_POLICY_V4.json')
    identity = dict(policy_id=policy['policy_id'], policy_sha256=sha256(ARTIFACTS / 'EVALUATION_POLICY_V4.json'))
    if (selection['status'] != 'COMPLETE' or selection['evaluation_policy'] != identity
            or selection['input_identities']['run_lock_sha256'] != sha256(ARTIFACTS / 'RUN_LOCK.json')
            or selection['input_identities']['results_sha256'] != sha256(ARTIFACTS / selection['input_identities']['results_snapshot'])
            or lock['data_manifest_sha256'] != sha256(ARTIFACTS / 'DATA_MANIFEST.json')
            or manifest['formal_population_sha256'] != sha256(assets['formal_population_path'])):
        raise ValueError('locked selection/data identity differs')
    endpoints = {'M0': 0, selection['control_star']['variant']: selection['control_star']['step'],
                 selection['candidate_star']['variant']: selection['candidate_star']['step']}
    if set(endpoints) != set(VARIANTS) or selection['seed'] != 146:
        raise ValueError('this diagnostic requires the actual locked M0/M3/M4 seed146 endpoints')
    sources = {**lock['numeric_sources'], **policy['policy_sources']}
    for relative, expected in sources.items():
        if sha256(PROJECT / relative) != expected:
            raise ValueError('frozen source identity differs: ' + relative)
    for relative in ('scripts/revisit_prompt_cooccurrence.py', 'models/task_memory_supervision.py',
                     'datasets/pointcept_utils.py', 'datasets/native_long_dataset.py',
                     'scripts/native_long_confirmation_inference.py', 'datasets/semseg.py'):
        path = PROJECT / relative
        if path.is_file():
            sources[relative] = sha256(path)
    bound_files = {name: sha256(ARTIFACTS / name) for name in (
        'RUN_LOCK.json', 'CAL_SELECTION_LOCK.json', 'EVALUATION_POLICY_V4.json', 'DATA_MANIFEST.json',
        selection['input_identities']['results_snapshot'])}
    evaluation = Path(assets['run_root']) / policy['namespace'] / 'evaluation'
    saved, evidence, input_files = {}, [], {}
    for role in ('SEL', 'PB'):
        records = manifest['records'][role]
        population = {record['input_id']: (record['reference_id'], record['horizon']) for record in records}
        if len(population) != len(records):
            raise ValueError('manifest input population has duplicates')
        for variant, step in endpoints.items():
            for repeat in range(3):
                folder = evaluation / f'{variant}-s146-u{step:06d}' / role / f'repeat{repeat}'
                result, status = read_json(folder / 'RESULTS.json'), read_json(folder / 'STATUS.json')
                request = result['request']
                if (result['status'] != status['status'] or status['status'] != 'COMPLETE'
                        or any(value != identity for value in (result['evaluation_policy'], status['evaluation_policy'], request['evaluation_policy']))
                        or any(request[key] != value or status[key] != value for key, value in dict(
                            variant=variant, step=step, adapter_seed=146, parent_seed=45, role=role,
                            repeat_index=repeat, numeric_sources=lock['numeric_sources'],
                            parent_sha256=lock['parent_sha256'], population_sha256=lock['data_manifest_sha256']).items())
                        or status['expected_inputs'] != status['completed_inputs']
                        or status['completed_inputs'] != manifest['counts'][role]):
                    raise ValueError('saved evaluation request/status/policy differs')
                rows = {row['input_id']: row for row in result['identity']}
                if (len(rows) != len(result['identity']) or
                        {key: (row['reference_id'], row['horizon']) for key, row in rows.items()} != population):
                    raise ValueError('saved input population differs from locked manifest')
                saved[role, variant, repeat] = {key: _saved_diagnostic(row) for key, row in rows.items()}
                input_files[folder / 'RESULTS.json'] = sha256(folder / 'RESULTS.json')
                input_files[folder / 'STATUS.json'] = sha256(folder / 'STATUS.json')
                evidence.append(dict(role=role, variant=variant, step=step, repeat_index=repeat,
                                     results_sha256=sha256(folder / 'RESULTS.json'),
                                     status_sha256=sha256(folder / 'STATUS.json')))
    valid_classes = yaml.safe_load(Path(assets['taxonomy_path']).read_text())['valid_class_ids']
    journal = Path(assets['run_root']) / 'statistics/same-class-diagnostics-20261016'
    journal.mkdir(parents=True, exist_ok=True)
    with (journal / 'EXECUTION.lock').open('a') as exclusive:
        fcntl.flock(exclusive, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.perf_counter()
        status = dict(status='RUNNING', pid=os.getpid(), inputs_completed=0,
                      inputs_total=sum(len(manifest['records'][role]) for role in ('SEL', 'PB')),
                      started_utc=datetime.now(timezone.utc).isoformat(), error=None)

        def heartbeat():
            status['heartbeat_utc'] = datetime.now(timezone.utc).isoformat()
            write_json_atomic(journal / 'STATUS.json', status)

        heartbeat()
        try:
            config, payload = parent_config(assets, variant='M0', seed=146)
            del payload
            loader = RevisitEpisodeLoader(config, assets['data_root'], read_json(assets['formal_population_path']))
            collation = loader.collate.base.collate_fn
            rows = []
            for role in ('SEL', 'PB'):
                for record in manifest['records'][role]:
                    spec = RevisitEpisodeSpec.from_record(record)
                    prefix = loader._sample(spec, 0, spec.horizon)
                    common = causal_prefix_truth(prefix[2], prefix[6][:, 3],
                        filter_out_classes=collation.filter_out_classes, label_offset=collation.label_offset,
                        ignore_class_threshold=collation.ignore_class_threshold, ambiguities=prefix[8])
                    current = common['temporal_stages'] == spec.horizon-1
                    present = common['masks'][:, current].any(1)
                    ids = common['ids'][present].tolist()
                    dataset = loader.dataset(spec.dataset_scope)
                    labels = dataset._remap_model_output(
                        common['labels'][present] + int(dataset.label_offset)).tolist()
                    truth_sha = hashlib.sha256(json.dumps(dict(ids=ids, labels=labels,
                        excluded_ambiguous_ids=sorted(ambiguous_identity_ids(prefix[8]))),
                                                         sort_keys=True).encode()).hexdigest()
                    for repeat in range(3):
                        matches = {}
                        for variant in VARIANTS:
                            population, matched = saved[role, variant, repeat][spec.input_id]
                            if set(ids) != population or len(ids) != len(population):
                                raise ValueError('CPU current GT differs from saved evaluation population')
                            matches[variant] = matched
                        counts = cooccurrence_counts(ids, labels, matches, ambiguities=prefix[8],
                                                     valid_class_ids=valid_classes)
                        rows.append(dict(role=role, repeat_index=repeat, input_id=spec.input_id,
                                         reference_id=spec.reference_id, horizon=spec.horizon,
                                         current_truth_metadata_sha256=truth_sha, **counts))
                    del prefix, common
                    status['inputs_completed'] += 1
                    if status['inputs_completed'] % 10 == 0:
                        heartbeat()
            if torch.cuda.is_initialized():
                raise ValueError('CUDA became initialized during offline diagnostics')
            if any(sha256(PROJECT / relative) != expected for relative, expected in sources.items()):
                raise ValueError('source changed during offline diagnostic')
            if any(sha256(ARTIFACTS / name) != expected for name, expected in bound_files.items()):
                raise ValueError('lock changed during offline diagnostic')
            if any(sha256(path) != expected for path, expected in input_files.items()):
                raise ValueError('saved evaluation input changed during offline diagnostic')
            groups = [dict(role=role, repeat_index=repeat, **_summary([
                row for row in rows if row['role'] == role and row['repeat_index'] == repeat]))
                for role in ('SEL', 'PB') for repeat in range(3)]
            artifact = dict(schema=1, status='COMPLETE', verified_utc=datetime.now(timezone.utc).isoformat(),
                evaluation_policy=identity, selected_endpoints=endpoints, adapter_seed=146, parent_seed=45,
                definition='Distinct reliable GT entities sharing an official class in the final current evaluable observation',
                matching=MATCHING, aggregation='Sum matched entities / sum eligible cooccurring entities within each process; zero denominator is null',
                scope='CURRENT_MASK_CONDITIONAL_RECALL_DESCRIPTIVE_ONLY_NOT_AP_OR_CAUSAL_INTERFERENCE',
                identity_exclusion='Negative unknown IDs and existing ambiguous_identity_ids groups; primary evaluation inputs/results unchanged',
                independent_references={role: manifest['independent_references'][role] for role in ('SEL', 'PB')},
                limitations=['Historically exposed panels; overlapping inputs are not independent scenes',
                             'Three inference processes share one adaptation seed and do not increase independent reference counts',
                             'Cooccurrence does not establish causal interference; no moving-object subset inferred'],
                valid_class_ids=valid_classes, source_sha256=sources, immutable_inputs=bound_files,
                taxonomy_sha256=sha256(assets['taxonomy_path']), inputs=evidence,
                inputs_completed=status['inputs_completed'], process_input_rows=len(rows),
                groups=groups, rows=rows, cuda_initialized=False, new_gpu_hours=0,
                elapsed_seconds=time.perf_counter()-started)
            write_json_atomic(output, artifact)
            status.update(status='COMPLETE', artifact_sha256=sha256(output),
                          process_input_rows=len(rows), elapsed_seconds=artifact['elapsed_seconds'])
            heartbeat()
            return dict(status='COMPLETE', artifact=output.name, artifact_sha256=status['artifact_sha256'],
                        inputs_completed=status['inputs_completed'], process_input_rows=len(rows), groups=groups)
        except BaseException as exc:
            status.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
            heartbeat()
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status='DRY_RUN_NOT_STARTED', scope='CPU_ONLY_SAVED_MATCHES_AND_CURRENT_GT')))
        return 0
    from scripts.revisit_prompt_assets import local_assets

    print(json.dumps(run_analysis(local_assets()), sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
