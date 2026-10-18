"""Independent OGR assets, recoverable budget journal and frozen numeric policy."""

from contextlib import contextmanager
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

from scripts.native_long_budget import merge_events, write_json_atomic
from scripts.revisit_prompt_assets import (
    ENCODER_PREFIXES, configure_runtime, digest, parent_config, read_json, sha256, validate_bound_file,
)


PROJECT = Path(__file__).resolve().parents[1]
ARTIFACTS = PROJECT / 'artifacts/observation_grounded_v1'
BASE = '758b5aa4bd2e9119ae5e05f725faa360c2120d88'
NEW_PREFIXES = ('model.mpr_prompt.', 'model.ogr_identity.', 'model.ogr_feedback.')
NUMERIC_POLICY = 'ogr-frozen-backbone-v4-derived-v1'
PHASE_CAPS = dict(D_DIAGNOSTIC=2., T_FIRST=16., T_REPLICATION=12.,
                  E_EVALUATION=12., F_PROFILE_CONFIRM=4., I_INTEGRATION_REPAIR=4.)
NEW_FILES = (
    'models/observation_evidence_state.py', 'models/observation_grounded_modules.py',
    'models/observation_grounded.py', 'models/observation_grounded_loss.py',
    'datasets/observation_grounded_episode.py', 'trainer/observation_grounded_trainer.py',
    'scripts/ogr_assets.py', 'scripts/ogr_runtime.py', 'scripts/ogr_evaluation.py',
    'scripts/ogr_diagnostics.py', 'scripts/ogr_selection.py', 'scripts/ogr_statistics.py',
    'scripts/ogr_campaign.py', 'scripts/ogr_delivery.py', 'scripts/ogr_policy.py',
    'conf/observation_grounded_v1/default.yaml',
)
PARENT_FILES = (
    'models/rescene.py', 'models/criterion.py', 'models/matcher.py', 'models/task_memory_routing.py',
    'models/task_memory_state.py', 'models/revisit_prompt.py', 'models/revisit_prompt_modules.py',
    'models/revisit_prompt_state.py', 'models/revisit_prompt_loss.py', 'models/revisit_prompt_numerics.py',
    'models/revisit_prompt_native_numerics.py', 'models/revisit_prompt_centroid_numerics.py',
    'models/revisit_prompt_sparse_reference.py', 'datasets/revisit_prompt_episode.py',
    'scripts/revisit_prompt_assets.py', 'scripts/revisit_prompt_runtime.py',
    'trainer/revisit_prompt_trainer.py', 'trainer/task_memory_trainer.py',
)


class BudgetJournal:
    """One flock-protected durable state; JSONL is a derived audit export.

    Reservations count once while active, then actual wall time replaces them.
    Reopening can never create a new run cap. GPU children have one card each.
    """
    def __init__(self, root, *, prior, external_reserved=0.):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'BUDGET_STATE.json'
        self.lock_path = self.root / '.ogr-budget.lock'
        identity = dict(prior_gpu_hours=float(prior), external_reserved_gpu_hours=float(external_reserved),
                        lifetime_cap_gpu_hours=1823., reserve_gpu_hours=8., run_cap_gpu_hours=50.,
                        phase_caps_gpu_hours=PHASE_CAPS)
        if not all(math.isfinite(v) and v >= 0 for v in (prior, external_reserved)):
            raise ValueError('invalid prior or external reservation')
        with self._locked():
            if self.path.exists():
                if read_json(self.path)['identity'] != identity:
                    raise ValueError('budget identity changed; explicit reconciliation is required')
            else:
                available = max(0., min(50., 1823. - float(prior) - float(external_reserved) - 8.))
                write_json_atomic(self.path, dict(schema=1, identity=identity,
                    initial_available_gpu_hours=available, events=[]))

    @contextmanager
    def _locked(self):
        with self.lock_path.open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @staticmethod
    def _snapshot(state):
        active = [event for event in state['events'] if event['status'] == 'ACTIVE']
        spent = sum(event['gpu_hours'] for event in state['events'] if event['status'] != 'ACTIVE')
        reserved = sum(event['reservation_gpu_hours'] for event in active)
        return dict(schema=1, identity=state['identity'],
            initial_available_gpu_hours=state['initial_available_gpu_hours'], spent_gpu_hours=spent,
            active_reserved_gpu_hours=reserved,
            remaining_gpu_hours=max(0., state['initial_available_gpu_hours'] - spent - reserved),
            active_jobs=active, event_count=len(state['events']))

    def snapshot(self):
        with self._locked():
            return self._snapshot(read_json(self.path))

    def export(self, destination):
        """Read and export under the same lock as settlement, including peers."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        with self._locked():
            state = read_json(self.path)
            snapshot = self._snapshot(state)
            write_json_atomic(destination / 'BUDGET_LOCK.json', snapshot)
            ledger = destination / 'COST_LEDGER.jsonl'
            temporary = ledger.with_suffix('.jsonl.tmp')
            temporary.write_text(''.join(json.dumps(row, sort_keys=True) + '\n'
                for row in state['events'] if row['status'] != 'ACTIVE'))
            os.replace(temporary, ledger)
            return snapshot

    def reserve(self, job_key, phase, gpu_hours, *, gpu):
        if phase not in PHASE_CAPS or not math.isfinite(gpu_hours) or gpu_hours <= 0:
            raise ValueError('invalid phase or GPU reservation')
        with self._locked():
            state = read_json(self.path)
            snapshot = self._snapshot(state)
            if any(event['job_key'] == job_key for event in state['events']):
                raise ValueError('job key already exists; retries need a documented distinct attempt')
            if (len(snapshot['active_jobs']) >= 2
                    or any(event['physical_gpu_index'] == gpu for event in snapshot['active_jobs'])):
                raise ValueError('at most two independent single-GPU workers, one per physical GPU')
            phase_used = sum((event['reservation_gpu_hours'] if event['status'] == 'ACTIVE'
                              else event['gpu_hours']) for event in state['events'] if event['phase'] == phase)
            if gpu_hours > snapshot['remaining_gpu_hours'] + 1e-12 or phase_used + gpu_hours > PHASE_CAPS[phase] + 1e-12:
                raise ValueError('run/lifetime/reserve or phase budget would be exceeded')
            event_id = f'ogr:{phase}:{uuid.uuid4().hex}'
            state['events'].append(dict(event_id=event_id, job_key=job_key, phase=phase,
                status='ACTIVE', started_unix=time.time(), physical_gpu_index=gpu,
                gpus=1, reservation_gpu_hours=float(gpu_hours)))
            write_json_atomic(self.path, state)
            return event_id

    def finish(self, event_id, *, seconds, exit_code):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('actual elapsed seconds must be finite and nonnegative')
        with self._locked():
            state = read_json(self.path)
            events = [event for event in state['events'] if event['event_id'] == event_id]
            if len(events) != 1 or events[0]['status'] != 'ACTIVE':
                raise ValueError('budget event is absent or already finalized')
            event = events[0]
            event.update(status='COMPLETE' if exit_code == 0 else 'FAILED', seconds=float(seconds),
                         gpu_hours=float(seconds) / 3600., exit_code=int(exit_code), ended_unix=time.time())
            write_json_atomic(self.path, state)
            ledger = self.root / 'COST_LEDGER.jsonl'
            temporary = ledger.with_suffix('.jsonl.tmp')
            temporary.write_text(''.join(json.dumps(row, sort_keys=True) + '\n'
                                         for row in state['events'] if row['status'] != 'ACTIVE'))
            os.replace(temporary, ledger)
            return self._snapshot(state)


def local_assets():
    return read_json(PROJECT / 'assets.local.json')


def numeric_sources(*, require_complete=True):
    missing = [name for name in (*PARENT_FILES, *NEW_FILES) if not (PROJECT / name).is_file()]
    if missing and require_complete:
        raise FileNotFoundError('OGR implementation not complete: ' + ', '.join(missing))
    return {name: sha256(PROJECT / name) for name in (*PARENT_FILES, *NEW_FILES) if (PROJECT / name).is_file()}


@contextmanager
def frozen_numeric_policy():
    """Restored patches; only frozen sparse backbone calls may use ordered_subm.

    This context does not disable autograd around the OGR decoder or pooling.
    The inherited model puts only its backbone call inside torch.no_grad().
    """
    from models.scatter import AdaptiveScatter
    from models.revisit_prompt_numerics import stable_group_mean
    from models.revisit_prompt_centroid_numerics import evaluation_centroids
    from models.revisit_prompt_sparse_reference import evaluation_sparse_reference

    original = AdaptiveScatter.mean_scatter
    AdaptiveScatter.mean_scatter = lambda self, x, index: stable_group_mean(x, index)
    try:
        with evaluation_centroids(), evaluation_sparse_reference():
            yield
    finally:
        AdaptiveScatter.mean_scatter = original


def load_parent_state(system, payload):
    import torch

    current = system.state_dict()
    parameters, buffers = dict(system.named_parameters()), dict(system.named_buffers())
    omitted = {key for key in parameters if key.startswith(ENCODER_PREFIXES)}
    encoder_buffers = {key for key in buffers if key.startswith(ENCODER_PREFIXES)}
    new = {key for key in current if key.startswith(NEW_PREFIXES)}
    expected_task = set(current) - omitted - new
    if (set(payload['omitted_encoder_parameter_names']) != omitted
            or set(payload['included_encoder_buffers']) != encoder_buffers
            or set(payload['state_dict']) != expected_task):
        raise ValueError('OGR parent task/buffer coverage differs; only declared new namespaces are allowed')
    for key, tensor in payload['state_dict'].items():
        if tensor.shape != current[key].shape or tensor.dtype != current[key].dtype:
            raise ValueError('parent tensor shape/dtype differs: ' + key)
        current[key] = tensor
    system.load_state_dict(current, strict=True)
    if any(not torch.equal(system.state_dict()[key].cpu(), value.cpu())
           for key, value in payload['state_dict'].items()):
        raise ValueError('loaded parent tensor bytes differ')
    return dict(status='EXACT_PARENT_TASK_AND_BUFFER_COVERAGE', loaded_task_entries=len(expected_task),
                omitted_encoder_entries=len(omitted), included_encoder_buffers=sorted(encoder_buffers),
                new_state_entries=sorted(new))


def build_system(assets, *, variant, seed):
    from datasets.native_long_dataset import isolated_rng
    from models.revisit_prompt_native_numerics import pin_native_convolutions
    from trainer.observation_grounded_trainer import ObservationGroundedTrainer

    config, payload = parent_config(assets, variant='M0' if variant in ('M0', 'FH-COMMIT0') else 'M4', seed=seed)
    config.model._target_ = 'models.observation_grounded.ObservationGroundedReScene'
    config.model.ogr_variant = 'M0' if variant == 'FH-COMMIT0' else variant
    with isolated_rng(seed):
        system = ObservationGroundedTrainer(config)
    audit = load_parent_state(system, payload)
    audit['native_convolutions'] = pin_native_convolutions(system)
    audit['trainable_parameters'] = sum(parameter.numel() for parameter in system.parameters() if parameter.requires_grad)
    expected = {'M0': 0, 'FH-COMMIT0': 0, 'M4': 84224, 'R-TARGET': 84224,
                'R-ID': 116993, 'R-QFB': 182785, 'R-OBS': 182785}
    if audit['trainable_parameters'] != expected[variant]:
        raise ValueError('OGR actual active parameter count differs')
    return system, config, audit


def parent_tensor_digest(system):
    import torch

    result = hashlib.sha256()
    for name, value in sorted(system.state_dict().items()):
        if name.startswith(NEW_PREFIXES):
            continue
        value = value.detach().cpu().contiguous()
        result.update(name.encode())
        result.update(str((tuple(value.shape), value.dtype)).encode())
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def load_ogr_bundle(path, *, assets=None):
    import torch
    from trainer.observation_grounded_trainer import load_adapter_state

    assets = local_assets() if assets is None else assets
    value = torch.load(path, map_location='cpu', weights_only=True)
    if (value.get('schema') != 1 or value.get('kind') != 'observation-grounded-delta'
            or value['parent_seed'] != 45
            or value['parent_sha256'] != assets['hashes']['parent']
            or value['encoder_sha256'] != assets['hashes']['encoder']
            or value['taxonomy_sha256'] != assets['hashes']['taxonomy']):
        raise ValueError('OGR public delta schema/parent/encoder/taxonomy identity differs')
    for name, expected in value['numeric_sources'].items():
        if sha256(PROJECT / name) != expected:
            raise ValueError('OGR public delta numeric source differs: ' + name)
    system, _, _ = build_system(assets, variant=value['variant'], seed=value['adapter_seed'])
    load_adapter_state(system, value['state_dict'])
    return system


def resolve_assets(config):
    """Reuse mounted, already referenced assets; derive a separate OGR run root."""
    legacy_root = PROJECT.parent / 'persist4d-mpr-v1-formal-20261014'
    previous = read_json(legacy_root / 'assets.local.json')
    previous_run = Path(previous['run_root'])
    if (previous_run / 'ACTIVE_PROCESS.json').exists() or (previous_run / 'ACTIVE_PROCESSES.json').exists():
        raise ValueError('previous campaign has an active reservation; reconcile its actual record first')
    ledger_path = previous_run / 'COST_LEDGER.jsonl'
    rows = merge_events([[json.loads(line) for line in ledger_path.read_text().splitlines() if line]])
    prior = previous['budget']['prior_gpu_hours'] + sum(row['gpu_hours'] for row in rows)
    if abs(prior - config['budget']['historical_reference_gpu_hours']) > .01:
        raise ValueError('actual previous ledger differs materially from supplied historical reference')
    run_root = PROJECT.parent / 'persist4d_ogr_v1_20261018_runs'
    assets = {key: previous[key] for key in ('data_root', 'encoder_path', 'parent_bundle',
        'taxonomy_path', 'population_path', 'formal_population_path', 'python', 'hashes', 'identity_cache')}
    assets.update(schema=1, repo_root=str(PROJECT), run_root=str(run_root), device='cuda:0',
                  previous_run_root=str(previous_run),
                  pythonpath=[str(PROJECT), *[p for p in previous['pythonpath'] if p != previous['repo_root']]],
                  budget=dict(prior_gpu_hours=prior, external_reserved_gpu_hours=0.,
                      prior_event_count=len(rows), prior_ledger_sha256=sha256(ledger_path)))
    cache = dict(assets['identity_cache'])
    for key, name in (('parent', 'parent_bundle'), ('encoder', 'encoder_path'), ('taxonomy', 'taxonomy_path'),
                      ('population', 'population_path'), ('formal_population', 'formal_population_path')):
        cache[assets[name]] = validate_bound_file(assets[name], assets['hashes'][key], cache)
    assets['identity_cache'] = cache
    if not Path(assets['data_root'], 'processed/rio/train_database.yaml').is_file():
        raise FileNotFoundError('mounted native rio database is unavailable')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=PROJECT, text=True).strip()
    subprocess.check_call(['git', 'merge-base', '--is-ancestor', BASE, 'HEAD'], cwd=PROJECT)
    if branch != config['branch']:
        raise ValueError('OGR must use its new dedicated branch')
    journal = BudgetJournal(run_root, prior=prior)
    # The resolver is private and is explicitly ignored, including in new worktrees.
    common = Path(subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], cwd=PROJECT, text=True).strip())
    if not common.is_absolute():
        common = (PROJECT / common).resolve()
    excluded = common / 'info/exclude'
    if '/assets.local.json' not in excluded.read_text().splitlines():
        with excluded.open('a') as stream:
            stream.write('\n/assets.local.json\n')
    write_json_atomic(PROJECT / 'assets.local.json', assets)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    population, formal = read_json(assets['population_path']), read_json(assets['formal_population_path'])
    records = {role: [{**record, 'role': role, 'dataset_scope': 'train'}
                      for record in population['evaluation_inputs'][role]] for role in ('CAL', 'SEL')}
    records['PB'] = [{**record, 'role': 'PB', 'dataset_scope': 'validation', 'scan_ids': record['physical_scan_ids']}
                     for record in formal['records']['PB']]
    counts = {role: dict(Counter(str(record['horizon']) for record in inputs)) for role, inputs in records.items()}
    expected = {role: {str(key): value for key, value in inputs.items()}
                for role, inputs in config['data']['expected_inputs'].items()}
    if counts != expected:
        raise ValueError('actual CAL/SEL/PB input counts differ from supplied contract')
    train_refs = {record['reference_id'] for record in population['references'] if record['role'] == 'TRAIN'}
    evaluation_refs = {record['reference_id'] for inputs in records.values() for record in inputs}
    if train_refs & evaluation_refs:
        raise ValueError('adaptation TRAIN overlaps held-out evaluation references')
    manifest = dict(schema=1, status='LOCKED_HISTORICALLY_EXPOSED', records=records,
        references=population['references'], counts=counts,
        stage_forwards={role: sum(record['horizon'] for record in inputs) for role, inputs in records.items()},
        independent_references={role: len({r['reference_id'] for r in inputs}) for role, inputs in records.items()},
        adaptation_train_overlap_count=0, source_population_sha256=assets['hashes']['population'],
        formal_population_sha256=assets['hashes']['formal_population'])
    write_json_atomic(ARTIFACTS / 'DATA_MANIFEST.json', manifest)
    reconcile_exposure(assets, population, formal, manifest)
    write_json_atomic(ARTIFACTS / 'BUDGET_LOCK.json', journal.snapshot())
    write_json_atomic(ARTIFACTS / 'START_STATE.json', dict(schema=1, status='ASSETS_AND_PRIOR_BOUND',
        base_commit=BASE, branch=branch, instruction_sha256=sha256(ARTIFACTS / 'EXECUTION_INSTRUCTION.md'),
        asset_hashes=assets['hashes'], parent_seed=45, adapter_seeds=[246, 247],
        data_counts=counts, budget=journal.snapshot(), source_sha256=numeric_sources(require_complete=False),
        missing_implementation_files=[name for name in NEW_FILES if not (PROJECT / name).exists()],
        publication_status='PENDING_EXTERNAL_VERIFICATION', local_paths_public=False))
    return assets


def reconcile_exposure(assets, population, formal, manifest):
    """Account for every known metadata reference; absence is not proof of no exposure."""
    import csv

    # Use the exact source already bound by FORMAL_POPULATION; no new data search.
    binding = formal['source_bindings']['metadata']
    metadata_path = Path(binding['path'])
    if sha256(metadata_path) != binding['sha256']:
        raise ValueError('known physical-reference metadata identity changed')
    metadata = read_json(metadata_path)
    roles = {row['reference_id']: row['role'] for row in population['references']}
    excluded = {row['reference_id']: row for row in population['excluded']}
    historical = {key: set(value) for key, value in population['historical_role_overlap'].items()}
    for role, coverage in formal['coverage'].items():
        for reference in coverage['reference_ids']:
            historical.setdefault(reference, set()).add('historical:'+role)
    for role, records in manifest['records'].items():
        for row in records:
            historical.setdefault(row['reference_id'], set()).add('historical:'+role)
    rows = []
    for item in sorted(metadata, key=lambda row: row['reference']):
        reference = item['reference']
        role = roles.get(reference)
        known_history = sorted(historical.get(reference, set()))
        reasons = excluded.get(reference, {}).get('reasons', [])
        if role == 'TRAIN':
            eligibility, reason = 'EXCLUDED', 'CURRENT_ADAPTER_TRAIN'
        elif role in ('CAL', 'SEL') or known_history:
            eligibility, reason = 'EXCLUDED', 'KNOWN_SELECTION_DIAGNOSTIC_OR_EVALUATION_EXPOSURE'
        else:
            eligibility, reason = 'UNKNOWN_EXPOSURE', 'NO_COMPLETE_TASK_EXPOSURE_ATTESTATION'
        rows.append(dict(reference_id=reference, official_split=item.get('type'), current_role=role,
            known_historical_roles=';'.join(known_history),
            known_parent_train_scope='UNKNOWN_UNLESS_IN_BOUND_TRAIN_POPULATION',
            current_adaptation_train=role == 'TRAIN',
            population_exclusion_reasons=';'.join(reasons),
            unavailable_scans=';'.join(excluded.get(reference, {}).get('unavailable_scans', [])),
            confirm_eligibility=eligibility, confirm_exclusion_reason=reason,
            encoder_pretraining_exposure='NOT_EXCLUDED_OR_ATTESTED'))
    with (ARTIFACTS / 'EXPOSURE_MAP.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    status = dict(schema=1, status='UNAVAILABLE', reference_count=0, records=[],
        metadata_references_checked=len(rows), metadata_sha256=binding['sha256'],
        current_train_references=sum(row['current_adaptation_train'] for row in rows),
        known_exposure_excluded=sum(row['confirm_eligibility'] == 'EXCLUDED' for row in rows),
        unknown_exposure_references=sum(row['confirm_eligibility'] == 'UNKNOWN_EXPOSURE' for row in rows),
        reason='No candidate has complete verified non-exposure to parent task training and all known historical research roles',
        unknown_is_independent=False, minimum_references=2, minimum_one_horizon=3,
        dataset_scope='EXPLICIT_ONLY_NO_PB_ALIAS', encoder_pretraining_non_exposure_claim=False,
        core_experiments_may_continue=True)
    write_json_atomic(ARTIFACTS / 'CONFIRM_STATUS.json', status)
    return status
