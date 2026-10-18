"""Export actual OGR evidence and publish immutable A/B commits and LFS bytes."""

import argparse
import ast
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

from scripts.native_long_budget import write_json_atomic
from scripts.ogr_assets import (
    ARTIFACTS, BASE, NEW_FILES, PARENT_FILES, PROJECT, local_assets,
    numeric_sources, read_json, sha256,
)
from scripts.ogr_diagnostics import write_csv
from scripts.ogr_selection import expected_evaluations, journal_for


TEST_FILES = tuple(f'tests/test_ogr_{name}.py' for name in ('data', 'learning', 'runtime', 'delivery'))
DOC_FILES = ('docs/superpowers/specs/2026-10-18-observation-grounded-design.md',
             'docs/superpowers/plans/2026-10-18-observation-grounded-plan.md')
EXCLUSIONS = {'ARTIFACT_MANIFEST.json', 'HANDOFF.md', 'PUBLICATION_RECEIPT.json'}


def git(*arguments, cwd=PROJECT, env=None):
    return subprocess.check_output(['git', *arguments], cwd=cwd, text=True,
        stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}).strip()


def publication_files():
    paths = {PROJECT/name for name in (*NEW_FILES, *TEST_FILES, *DOC_FILES, '.gitattributes')}
    paths.update(path for path in ARTIFACTS.rglob('*') if path.is_file() and path.name not in EXCLUSIONS)
    return sorted(paths)


def public_record(value, assets):
    """Redact private mounted paths in copied logs, without changing source files."""
    if isinstance(value, dict):
        return {key: public_record(item, assets) for key, item in value.items()}
    if isinstance(value, list):
        return [public_record(item, assets) for item in value]
    if isinstance(value, str):
        replacements = {assets[key]: f'<{key}>' for key in ('data_root', 'encoder_path', 'parent_bundle',
            'taxonomy_path', 'population_path', 'formal_population_path', 'run_root', 'repo_root', 'previous_run_root')}
        for path, replacement in sorted(replacements.items(), key=lambda row: -len(row[0])):
            value = value.replace(path, replacement)
        value = value.replace(str(Path.home()), '<HOME>')
    return value


def require_settled(assets):
    snapshot = journal_for(assets).export(ARTIFACTS)
    if snapshot['active_jobs']:
        raise ValueError('all OGR workers and accounting reservations must settle before delivery')
    root = Path(assets['run_root'])
    process_file = root / 'PROCESS_EVENTS.jsonl'
    for row in [json.loads(line) for line in process_file.read_text().splitlines() if line] if process_file.exists() else []:
        proc = Path('/proc') / str(row['pid'])
        if proc.exists():
            try:
                command = (proc / 'cmdline').read_bytes()
                cwd = (proc / 'cwd').resolve()
            except (FileNotFoundError, PermissionError):
                continue
            if b'scripts.ogr_campaign' in command and cwd == PROJECT:
                raise ValueError('an OGR child is still alive after accounting settlement')
    return snapshot


def export_sources():
    rows = []
    for name in (*PARENT_FILES, *NEW_FILES, *TEST_FILES):
        path = PROJECT / name
        symbols = []
        if path.suffix == '.py':
            tree = ast.parse(path.read_text())
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    symbols.append(dict(symbol=node.name, line=node.lineno))
                    if isinstance(node, ast.ClassDef):
                        symbols.extend(dict(symbol=node.name+'.'+item.name, line=item.lineno)
                            for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)))
        tracked = subprocess.run(['git', 'cat-file', '-e', f'{BASE}:{name}'], cwd=PROJECT, capture_output=True).returncode == 0
        rows.append(dict(path=name, sha256=sha256(path), status='UNCHANGED_PARENT' if tracked else 'NEW_OGR_FILE', symbols=symbols))
    result = dict(schema=1, base_commit=BASE, files=rows,
        numeric_sources=numeric_sources(), parent_files_modified=[],
        diff_scope='new OGR modules/config/tests/docs and task-scoped LFS attributes; original MPR implementation preserved')
    write_json_atomic(ARTIFACTS / 'SOURCE_BINDINGS.json', result)
    return result


def export_training(assets):
    root = Path(assets['run_root'])
    rows, stages = [], []
    for group in ('integration', 'probe', 'trajectories'):
        for path in sorted((root / group).glob('*/TRAINING.jsonl')):
            for line in path.read_text().splitlines():
                record = json.loads(line)
                rows.append(dict(mode=record['mode'], variant=record['variant'], adapter_seed=record['adapter_seed'],
                    step=record['step'], loss=record['loss'], seconds=record['seconds'], lr=record['lr'],
                    episode_count=record['episode_count'], stage_count=record['stage_count'],
                    selected_slots=sum(stage['selected_slots'] for stage in record['stages']),
                    bound_births=sum(stage['birth_labels']['bound_births'] for stage in record['stages']),
                    forced_targets=sum(stage['supervision']['forced'] for stage in record['stages']),
                    event_id=record['event_id'], evidence=path.relative_to(root).as_posix()))
                for stage in record['stages']:
                    stages.append(dict(mode=record['mode'], variant=record['variant'], adapter_seed=record['adapter_seed'],
                        step=record['step'], event_id=record['event_id'], **stage))
    write_csv(ARTIFACTS / 'TRAINING.csv', rows, fieldnames=None if rows else ['mode', 'variant', 'step', 'status'])
    path = ARTIFACTS / 'evidence/TRAINING_STAGES.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, sort_keys=True, allow_nan=False)+'\n' for row in stages))
    audits = []
    for path in sorted((root / 'probe').glob('*/STATUS.json')):
        value = read_json(path)
        audits.append(dict(variant=value['variant'], step=value['step'], status=value['status'],
            load_audit=value['load_audit'], frozen_parent_unchanged=value['frozen_parent_unchanged'],
            parent_tensor_sha256=value['parent_tensor_sha256'], changed_new_tensors=value['trainable_tensor_changed']))
    write_json_atomic(ARTIFACTS / 'PARAMETER_AND_LOAD_AUDIT.json', dict(status='ACTUAL_20_UPDATE_PROBES', variants=audits,
        integration=read_json(ARTIFACTS / 'INTEGRATION_VERIFICATION.json')))
    return rows


def export_models(assets, plan):
    root = Path(assets['run_root'])
    rows, missing, aliases = [], [], []
    import torch
    for group, destination in (('integration', 'integration'), ('probe', 'probe'), ('trajectories', 'resume')):
        for path in sorted((root / group).glob('*/resume-latest.pt')):
            value = torch.load(path, map_location='cpu', weights_only=False)
            identity = value['identity']
            name = f'{path.parent.name}-u{value["step"]:06d}.resume.pt'
            target = ARTIFACTS / destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != path.stat().st_size:
                shutil.copyfile(path, target)
            rows.append(dict(path=target.relative_to(PROJECT).as_posix(), bytes=target.stat().st_size, sha256=sha256(target),
                kind='formal_resume' if group == 'trajectories' else group+'_resume',
                variant=identity['variant'], adapter_seed=identity['adapter_seed'], step=value['step'],
                formal_model=group == 'trajectories', optimizer_included=True,
                training_identity=identity, source_relative_to_run=path.relative_to(root).as_posix()))
    for path in sorted((root / 'trajectories').glob('*/delta-*.pt')):
        value = torch.load(path, map_location='cpu', weights_only=True)
        target = ARTIFACTS / 'deployment' / f'{path.parent.name}-{path.name}'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        rows.append(dict(path=target.relative_to(PROJECT).as_posix(), bytes=target.stat().st_size, sha256=sha256(target),
            kind='formal_delta', variant=value['variant'], adapter_seed=value['adapter_seed'], step=value['step'],
            formal_model=True, optimizer_included=False))
    for seed in (246, 247):
        lock_path = ARTIFACTS / ('CAL_SELECTION_LOCK.json' if seed == 246 else 'REPLICATION_LOCK.json')
        lock = read_json(lock_path) if lock_path.is_file() else {}
        for variant, endpoint in lock.get('endpoints', {}).items():
            step = endpoint['point']['step']
            item = next(row for row in rows if row['kind'] == 'formal_delta' and
                        (row['variant'], row['adapter_seed'], row['step']) == (variant, seed, step))
            aliases.append(dict(alias=f'{variant}-s{seed}-CAL-selected', path=item['path'], sha256=item['sha256'],
                                eligible=endpoint['eligible']))
    if plan['N'] is None:
        missing.append(dict(kind='formal_delta', expected_if_feasible=24, actual=0, reason='NO_FEASIBLE_COMMON_N'))
        missing.append(dict(kind='formal_resume', expected_if_feasible=6, actual=0, reason='NO_FEASIBLE_COMMON_N'))
    result = dict(schema=1, assets=rows, selected_aliases=aliases, missing=missing,
        formal_delta_count=sum(row['kind'] == 'formal_delta' for row in rows),
        formal_resume_count=sum(row['kind'] == 'formal_resume' for row in rows),
        probe_resumes_are_selected_formal_models=False,
        parent_sha256=assets['hashes']['parent'], encoder_sha256=assets['hashes']['encoder'],
        taxonomy_sha256=assets['hashes']['taxonomy'])
    write_json_atomic(ARTIFACTS / 'MODEL_MANIFEST.json', result)
    return result


def export_official_states(assets, models):
    root = Path(assets['run_root'])
    members = []
    path = ARTIFACTS / 'evidence/official_states.zip'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        sources = list((root / 'evaluation').glob('*/*/official-*.json'))
        sources += list((root / 'evaluation').glob('*/*/references/*.json'))
        sources += list((root / 'numeric').glob('*/*/*.official.json'))
        for source in sorted(sources):
            value = read_json(source)
            if value.get('schema_version') != 1 or not value.get('states'):
                raise ValueError('archive member must be an actual official sufficient state')
            relative = source.relative_to(root).as_posix()
            archive.write(source, relative)
            members.append(dict(member=relative, bytes=source.stat().st_size, sha256=sha256(source),
                mode=value['mode'], updates=value['updates'], formal=relative.startswith('evaluation/')))
    write_json_atomic(ARTIFACTS / 'OFFICIAL_STATE_INDEX.json', dict(member_count=len(members), members=members,
        empty_reason=None if members else 'No formal evaluation or finite-panel official states were produced'))
    models['assets'].append(dict(path=path.relative_to(PROJECT).as_posix(), kind='official_states_archive',
        bytes=path.stat().st_size, sha256=sha256(path), member_count=len(members), formal_model=False))
    write_json_atomic(ARTIFACTS / 'MODEL_MANIFEST.json', models)
    return members


def export_evaluations(assets, plan):
    root = Path(assets['run_root'])
    complete, references, identities, mask_rows = [], [], [], []
    for path in sorted((root / 'evaluation').glob('*/*/RESULTS.json')):
        result = read_json(path)
        if result['status'] != 'COMPLETE':
            continue
        complete.extend(result['rows'])
        references.extend(dict(**row, **result['request']) for row in result['references'])
        for episode in result['episodes']:
            identities.append(dict(**episode['identity'], input_id=episode['input_id'], **result['request']))
            for row in episode['current_mask_diagnostics']:
                mask_rows.append(dict(**row, input_id=episode['input_id'], **result['request']))
    first = read_json(ARTIFACTS / 'CAL_SELECTION_LOCK.json') if (ARTIFACTS / 'CAL_SELECTION_LOCK.json').exists() else None
    second = read_json(ARTIFACTS / 'REPLICATION_LOCK.json') if (ARTIFACTS / 'REPLICATION_LOCK.json').exists() else None
    expected = expected_evaluations(assets, plan, first, second)
    for job in expected['jobs']:
        method = 'M4' if job['method'] == 'M4-200' else job['method']
        seed = job['seed'] or 246
        matching = [row for row in complete if (row['variant'], row['adapter_seed'], row['step'], row['role']) ==
                    (method, seed, job['actual_step'], job['role'])]
        if matching:
            job['status'] = 'COMPLETE'
            job['completed_horizons'] = [row['horizon'] for row in matching]
    expected['complete_jobs'] = sum(job['status'] == 'COMPLETE' for job in expected['jobs'])
    write_json_atomic(ARTIFACTS / 'EXPECTED_EVALUATIONS.json', expected)
    not_run = dict(status='BUDGET_BLOCKED' if plan['N'] is None else 'INCOMPLETE', reason='NO_FEASIBLE_COMMON_N' if plan['N'] is None else 'MISSING_EVIDENCE')
    write_csv(ARTIFACTS / 'CAL_ALL.csv', [row for row in complete if row['role'] == 'CAL'] or [not_run])
    write_csv(ARTIFACTS / 'MAIN_RESULTS.csv', [row for row in complete if row['role'] in ('SEL', 'PB', 'CONFIRM')] or [not_run])
    write_csv(ARTIFACTS / 'BY_REFERENCE.csv', references or [not_run])
    write_csv(ARTIFACTS / 'IDENTITY_COUNTS.csv', identities or [not_run])
    baseline_masks = {(row['role'], row['input_id'], row['stage']): row for row in mask_rows if row['variant'] == 'M0'}
    gain_loss = []
    for row in mask_rows:
        if row['variant'] == 'M0':
            continue
        baseline = baseline_masks.get((row['role'], row['input_id'], row['stage']))
        if baseline is None:
            raise ValueError('mask gain/loss requires the same actual M0 target event')
        base_targets = set(baseline['matched_gt_ids']) | set(baseline['unmatched_gt_ids'])
        target_ids = set(row['matched_gt_ids']) | set(row['unmatched_gt_ids'])
        if base_targets != target_ids:
            raise ValueError('mask gain/loss target event denominator differs')
        before, after = set(baseline['matched_gt_ids']), set(row['matched_gt_ids'])
        gained, lost = sorted(after-before), sorted(before-after)
        gain_loss.append(dict(variant=row['variant'], adapter_seed=row['adapter_seed'], step=row['step'],
            role=row['role'], input_id=row['input_id'], stage=row['stage'],
            gt_count=len(target_ids), M0_matched=len(before), method_matched=len(after),
            recovered_gt_ids=gained, lost_gt_ids=lost, recovered=len(gained), lost=len(lost),
            net_recovered=len(gained)-len(lost), includes_ephemeral_candidates=True))
    write_csv(ARTIFACTS / 'MASK_GAIN_LOSS.csv', gain_loss or [not_run])
    mechanism = root / 'mechanism'
    for source, destination in (('TARGET_FLOW.csv', 'SHARED_HISTORY_TARGET_FLOW.csv'),
                                ('TARGET_INTERVENTIONS.csv', 'TARGET_INTERVENTIONS.csv')):
        path = mechanism / source
        if path.is_file():
            with path.open(newline='') as stream:
                write_csv(ARTIFACTS / destination, list(csv.DictReader(stream)))
    interventions = mechanism / 'TARGET_INTERVENTIONS.csv'
    if interventions.is_file():
        with interventions.open(newline='') as stream:
            same_class = [row for row in csv.DictReader(stream) if row.get('intervention') == 'SAME_CLASS_SOURCE_SWAP']
        write_csv(ARTIFACTS / 'SAME_CLASS.csv', same_class or [dict(status='NA_NO_RELIABLE_SAME_CLASS_PAIR')])
    for name in ('TARGET_INTERVENTIONS.csv', 'SAME_CLASS.csv', 'PROFILE.csv'):
        if not (ARTIFACTS / name).exists():
            write_csv(ARTIFACTS / name, [dict(status='NOT_RUN_NO_FORMAL_ENDPOINTS', reason=not_run['reason'])])
    for name in ('CAL_SELECTION_LOCK.json', 'REPLICATION_LOCK.json'):
        if not (ARTIFACTS / name).exists():
            write_json_atomic(ARTIFACTS / name, dict(schema=1, **not_run, endpoints={},
                selected_endpoint=None, strongest_control=None, test_reselection=False))
    return expected


def export_execution(assets):
    root = Path(assets['run_root'])
    state = read_json(root / 'BUDGET_STATE.json')
    process_rows = [json.loads(line) for line in (root / 'PROCESS_EVENTS.jsonl').read_text().splitlines() if line]
    processes = {row['event_id']: row for row in process_rows}
    commands = []
    for row in state['events']:
        process = processes.get(row['event_id'], {})
        commands.append(dict(job_key=row['job_key'], event_id=row['event_id'], status=row['status'],
            exit_code=row.get('exit_code'), physical_gpu_index=row['physical_gpu_index'],
            seconds=row.get('seconds'), gpu_hours=row.get('gpu_hours'), stdout=process.get('log'),
            parameters=process.get('worker_arguments', 'see job_key and frozen campaign dispatcher for historical launches')))
    write_json_atomic(ARTIFACTS / 'COMMANDS_EXECUTED.json', dict(commands=commands, scope='actual charged process launches'))
    evidence = ARTIFACTS / 'evidence'
    for path in sorted((root / 'logs').glob('*.log')):
        (evidence / path.name).write_text(public_record(path.read_text(), assets))
    for name, pattern in (('PROBE_STATUS.json', 'probe/*/STATUS.json'), ('TIMING_STATUS.json', 'timing/*/*/STATUS.json')):
        write_json_atomic(evidence / name, public_record([dict(path=path.relative_to(root).as_posix(), **read_json(path))
                                                        for path in sorted(root.glob(pattern))], assets))
    for group in ('diagnostics', 'numeric', 'repairs'):
        for path in sorted((root / group).rglob('*.json')):
            target = evidence / path.relative_to(root)
            write_json_atomic(target, public_record(read_json(path), assets))
    (ARTIFACTS / 'COMMANDS.md').write_text(
        '# OGR actual execution and reproduction\n\n'
        'Actual charged children, physical cards, stdout paths and exit codes: COMMANDS_EXECUTED.json. '
        'The evidence directory preserves failed and successful attempts. Commands below are entry points; '
        'a listed entry point alone does not assert execution.\n\n```sh\n'
        'python -m scripts.ogr_campaign resolve --config conf/observation_grounded_v1/default.yaml\n'
        'python -m scripts.ogr_campaign diagnose --config conf/observation_grounded_v1/default.yaml\n'
        'python -m scripts.ogr_campaign probe --config conf/observation_grounded_v1/default.yaml\n'
        'python -m scripts.ogr_campaign lock --config conf/observation_grounded_v1/default.yaml\n'
        'python -m scripts.ogr_campaign run --config conf/observation_grounded_v1/default.yaml\n'
        'python -m scripts.ogr_campaign finalize --config conf/observation_grounded_v1/default.yaml\n'
        'python -m scripts.ogr_delivery publish --config conf/observation_grounded_v1/default.yaml\n```\n\n'
        'A fixed-retry entry was executed only after its specific regression and preserved repair receipt. '
        'No formal endpoint or 44-job completion is implied by the probe records.\n')


def finalize(assets, config):
    budget = require_settled(assets)
    plan = read_json(ARTIFACTS / 'RUN_PLAN.json')
    sources = export_sources()
    training = export_training(assets)
    models = export_models(assets, plan)
    members = export_official_states(assets, models)
    expected = export_evaluations(assets, plan)
    export_execution(assets)
    from scripts.ogr_statistics import compute_statistics
    statistics_result = compute_statistics(assets)
    policy = dict(policy_id='ogr-frozen-backbone-v4-derived-v1', frozen_backbone_only_no_grad=True,
        train_inference_mode=False, backbone_ordered_subm=True, dense_scatter='stable_group_mean',
        centroids='stable_segment_centroids', precision='32-true', TF32=False,
        inherited_internal_attention='Concerto FP16 FlashAttention where required',
        global_determinism_claim=False, numeric_sources=sources['numeric_sources'],
        new_path_panel=read_json(ARTIFACTS / 'NEW_PATH_NUMERIC_PANEL.json') if (ARTIFACTS / 'NEW_PATH_NUMERIC_PANEL.json').is_file() else {'status': 'NOT_RUN'},
        final_endpoint_panel=read_json(ARTIFACTS / 'FINAL_NUMERIC_PANEL.json') if (ARTIFACTS / 'FINAL_NUMERIC_PANEL.json').is_file() else {'status': 'NOT_RUN_NO_FORMAL_ENDPOINTS'})
    write_json_atomic(ARTIFACTS / 'NUMERIC_POLICY.json', policy)
    formal_updates = sum(row['mode'] == 'formal' for row in training)
    total = assets['budget']['prior_gpu_hours']+budget['spent_gpu_hours']
    status = dict(schema=1, CODE='IMPLEMENTED_FOCUSED_VALIDATION',
        EXPERIMENT='BUDGET_BLOCKED' if plan['N'] is None else 'CORE_COMPLETE' if expected['complete_jobs'] == 44 else 'INCOMPLETE',
        SCIENCE='NOT_TESTED' if plan['N'] is None else statistics_result['science']['PRIMARY'],
        PUBLICATION='PENDING_EXTERNAL_VERIFICATION', formal_N=plan['N'], formal_updates=formal_updates,
        completed_core_evaluations=expected['complete_jobs'], expected_core_evaluations=44,
        formal_deltas=models['formal_delta_count'], official_state_members=len(members),
        probe_updates=80, parent_seed=45, adapter_seeds=[246, 247],
        spent_gpu_hours=budget['spent_gpu_hours'], prior_gpu_hours=assets['budget']['prior_gpu_hours'],
        cumulative_gpu_hours=total, lifetime_remaining_gpu_hours=1823.-total,
        protected_reserve_gpu_hours=8., active_jobs=0, primary_review_complete=False)
    write_json_atomic(ARTIFACTS / 'STATUS.json', status)
    write_reports(assets, config, status, plan, models)
    return status


def write_reports(assets, config, status, plan, models):
    diagnostic = Path(assets['run_root']) / 'diagnostics/RESULTS.json'
    p1 = read_json(diagnostic) if diagnostic.is_file() else {'status': 'NOT_RUN', 'completed': []}
    forecast = plan['evaluation_forecast']['gpu_hours']
    (ARTIFACTS / 'REPORT.md').write_text(
        '# OGR-V1 evidence report\n\n'
        f'CODE: {status["CODE"]}; EXPERIMENT: {status["EXPERIMENT"]}; SCIENCE: {status["SCIENCE"]}; '
        'PUBLICATION: pending external receipt.\n\n'
        'The common-N gate uses actual four-arm, 20-update TRAIN probes and score-free CAL/PB structural timing. '
        f'The full 44-job evaluation forecast is {forecast:.6f} GPUh against the fixed 12 GPUh evaluation cap. '
        f'The declared result is {plan["status"]}; formal N is {plan["N"]}. '
        'No probe is relabelled a formal experiment, no guard winner was selected, and seed247 was not trained. '
        'The second seed is blocked by the common campaign gate, not cancelled because of negative first-seed scores.\n\n'
        'R-TARGET uses directly constrained task assignment; R-ID adds regional identity supervision and association; '
        'R-QFB and R-OBS add equal-size feedback, differing only in the direct value (normalized query versus current regional pool). '
        'Actual active counts are 84,224 / 116,993 / 182,785 / 182,785. '
        'The E1-23760 seed45 parent parameters and buffers were checked unchanged after real updates. '
        'All adaptation seeds share that parent; they are not independent parent retrainings.\n\n'
        f'P1 status: {p1["status"]}; actual method/input terminal evaluations: {len(p1.get("completed", []))}/34. '
        'TARGET_FLOW and ANCHOR_REFERENCE describe original M4-200 with shared M0 prediction history. '
        'Anchor interventions are GT_ASSISTED_DIAGNOSTIC, not official scores, upper bounds, or formal OGR comparisons. '
        'Reliable current/positive-segment/canonical-entity denominators are separate. '
        'Published target IoU is tied to source_query_ids and the correct class, including zero for a missing valid prediction.\n\n'
        'CAL/SEL/PB preserve their original complete input sets and are historically exposed. '
        'CONFIRM is unavailable because complete non-exposure to parent training and prior research could not be verified. '
        'UNKNOWN_EXPOSURE is not independence. Encoder pretraining exposure is not excluded. '
        'No moving-object subset or physical-day interpretation is invented.\n\n'
        'Formal CAL, SEL/PB quality, current AP gains, identity improvements, recovery/loss differences, '
        'same-class intervention differences, two-seed differences and reference-bootstrap intervals are NA. '
        'MAIN_RESULTS, CAL_ALL and BOOTSTRAP_RESULTS retain their blocked states. '
        'No previous MPR positive result is substituted for a new OGR result. '
        'MECHANISM_SUPPORTED_INTERNAL and CVPR readiness are not established.\n\n'
        f'Actual new cost: {status["spent_gpu_hours"]:.9f} GPUh; cumulative: {status["cumulative_gpu_hours"]:.9f} GPUh; '
        f'lifetime remainder: {status["lifetime_remaining_gpu_hours"]:.9f} GPUh, including the protected 8 GPUh. '
        'Startup, failed integration, fixed retries, probes and structural timing are charged once in COST_LEDGER. '
        'At most two one-card children ran concurrently. Unrelated jobs were not stopped.\n\n'
        f'Actual formal delta count: {models["formal_delta_count"]}; final formal resumes: {models["formal_resume_count"]}. '
        'Available probe/integration resumes are labelled nonformal. The manifest enumerates exact bytes; '
        'no missing checkpoint is fabricated. FINAL_REVIEW records the primary content audit. '
        'The immutable A/B publication and external receipt establish remote Git/LFS facts separately from research success.\n')
    (ARTIFACTS / 'DEPLOYMENT.md').write_text(
        '# OGR deployment contract\n\n'
        f'Parent: artifacts/native_long_postselection_v1/deployment/E1-23760.pt; SHA256 {assets["hashes"]["parent"]}.\n\n'
        f'Encoder SHA256: {assets["hashes"]["encoder"]}; taxonomy SHA256: {assets["hashes"]["taxonomy"]}. '
        'Obtain licensed 3RScan/Concerto assets through the original project instructions; they are not repackaged. '
        'Bind the existing authorized data and asset paths in the private, ignored assets.local.json.\n\n'
        'A formal deployment requires a manifest formal_delta, the fixed parent and encoder. '
        'scripts.ogr_assets.load_ogr_bundle performs strict source/parent/tensor checks. '
        'Run prediction inside frozen_numeric_policy and torch.no_grad; training retains decoder/pool autograd. '
        'Start each episode with ObservationEpisodeRuntime empty state. Use original official candidate extraction, '
        'source_query_ids, current-only commit0, and mean trajectory scores. Identity P never scales scores.\n\n'
        'There is no selected formal deployment in a NO_FEASIBLE_COMMON_N release. Probe resumes are reproducibility '
        'assets, not validated endpoints. Exact optimizer recovery uses restore_resume with the original identity, '
        'scheduler boundary and next_draw=4*step; never load probe optimizer state into a formal run.\n')


def validate_manifest_assets(manifest):
    for row in manifest['assets']:
        path = PROJECT / row['path']
        if not path.resolve().is_relative_to(ARTIFACTS.resolve()):
            raise ValueError('publication entity escapes the task artifact root')
        if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError('publication entity identity differs: '+row['path'])
    return manifest['assets']


def verify_remote_references(references, *, branch, tag, commit_b):
    if any(references.get(name) != commit_b for name in (f'refs/heads/{branch}', f'refs/tags/{tag}^{{}}')):
        raise ValueError('remote branch or peeled annotated tag differs from B')
    tag_object = references.get(f'refs/tags/{tag}')
    if not tag_object or tag_object == commit_b:
        raise ValueError('a present annotated tag object is required for publication')
    return True


def publish(assets, config):
    budget = require_settled(assets)
    status = read_json(ARTIFACTS / 'STATUS.json')
    if not status.get('primary_review_complete') or not (ARTIFACTS / 'FINAL_REVIEW.md').is_file():
        raise ValueError('publish requires the actual completed primary content review')
    branch, tag = config['branch'], config['tag']
    if git('branch', '--show-current') != branch:
        raise ValueError('publish must use the dedicated task branch')
    origin = git('remote', 'get-url', 'origin')
    if origin not in ('git@github.com:Orangekostar/Persist4D.git', 'https://github.com/Orangekostar/Persist4D.git'):
        raise ValueError('publication origin must be Orangekostar/Persist4D')
    models = read_json(ARTIFACTS / 'MODEL_MANIFEST.json')
    rows = validate_manifest_assets(models)
    root = Path(assets['run_root']) / 'publication'
    root.mkdir(parents=True, exist_ok=True)
    pending_path = root / 'PUBLICATION_PENDING.json'
    if pending_path.is_file():
        pending = read_json(pending_path)
        if (pending['branch'], pending['tag']) != (branch, tag) or git('rev-parse', 'HEAD') != pending.get('commit_b', pending['commit_a']):
            raise ValueError('publication recovery must preserve the exact A/B identity')
    else:
        git('lfs', 'install', '--local')
        files = publication_files()
        frozen = [dict(path=path.relative_to(PROJECT).as_posix(), bytes=path.stat().st_size, sha256=sha256(path)) for path in files]
        write_json_atomic(ARTIFACTS / 'ARTIFACT_MANIFEST.json', dict(schema=1, files=frozen,
            excludes=sorted(EXCLUSIONS), self_sha_embedded=False, future_A_or_B_SHA_embedded=False))
        git('add', '-f', '--', *[path.relative_to(PROJECT).as_posix() for path in files],
            (ARTIFACTS / 'ARTIFACT_MANIFEST.json').relative_to(PROJECT).as_posix())
        git('diff', '--cached', '--check')
        git('commit', '-m', 'Implement OGR four-arm runtime and publish measured '+status['EXPERIMENT'].lower()+' evidence')
        pending = dict(commit_a=git('rev-parse', 'HEAD'), branch=branch, tag=tag)
        write_json_atomic(pending_path, pending)
    if 'commit_b' not in pending:
        handoff = ('# OGR handoff\n\n'
            f'Experiment/code commit A: `{pending["commit_a"]}`.\n\n'
            f'Branch: `{branch}`; annotated tag: `{tag}`. The branch and peeled tag target the handoff commit; '
            'this document does not embed its own commit SHA.\n\n'
            f'CODE={status["CODE"]}; EXPERIMENT={status["EXPERIMENT"]}; SCIENCE={status["SCIENCE"]}. '
            'Publication facts are in the external receipt, not a self-modifying tag.\n\n'
            f'Formal N={status["formal_N"]}; core evaluations={status["completed_core_evaluations"]}/44; '
            f'formal deltas={status["formal_deltas"]}. No formal endpoint or strongest control was selected.\n\n'
            f'New cost={status["spent_gpu_hours"]:.9f} GPUh; cumulative={status["cumulative_gpu_hours"]:.9f} GPUh. '
            'The protected 8 GPUh remains available.\n\n'
            f'Parent SHA256: {assets["hashes"]["parent"]}; encoder SHA256: {assets["hashes"]["encoder"]}. '
            'REPORT.md, DEPLOYMENT.md, COMMANDS.md, MODEL_MANIFEST.json and FINAL_REVIEW.md define the actual scope. '
            'Probe/integration resumes are explicitly nonformal; missing formal checkpoints are not replaced. '
            'Reproduction uses the frozen source/asset identities and the original authorized native data.\n\n'
            'The run-root publication/PUBLICATION_RECEIPT.json records actual downloads, bytes, SHA256, remote refs '
            'and Release API availability. Git/LFS success and Release availability are separate statuses.\n')
        (ARTIFACTS / 'HANDOFF.md').write_text(handoff)
        git('add', '--', (ARTIFACTS / 'HANDOFF.md').relative_to(PROJECT).as_posix())
        git('commit', '-m', 'Add OGR handoff referencing frozen experiment commit A')
        pending['commit_b'] = git('rev-parse', 'HEAD')
        write_json_atomic(pending_path, pending)
    if not git('tag', '--list', tag):
        git('tag', '-a', tag, pending['commit_b'], '-m', 'OGR-V1 actual probes, diagnostics and budget-bound handoff')
    elif git('cat-file', '-t', f'refs/tags/{tag}') != 'tag' or git('rev-parse', f'{tag}^{{}}') != pending['commit_b']:
        raise ValueError('existing tag is not the required annotated tag pointing to B; it will not be moved')
    receipt_path = root / 'PUBLICATION_RECEIPT.json'
    receipt = dict(schema=1, **pending, timestamp_utc=datetime.now(timezone.utc).isoformat(),
        GIT_LFS='INCOMPLETE', RELEASE='NOT_ATTEMPTED', assets=[],
        spent_gpu_hours=budget['spent_gpu_hours'], cumulative_gpu_hours=status['cumulative_gpu_hours'])
    try:
        if rows:
            git('lfs', 'push', '--object-id', 'origin', *sorted({row['sha256'] for row in rows}))
        git('push', 'origin', f'HEAD:refs/heads/{branch}', f'refs/tags/{tag}', env={'GIT_LFS_SKIP_PUSH': '1'})
        output = git('ls-remote', 'origin', f'refs/heads/{branch}', f'refs/tags/{tag}', f'refs/tags/{tag}^{{}}')
        references = {line.split()[1]: line.split()[0] for line in output.splitlines()}
        verify_remote_references(references, branch=branch, tag=tag, commit_b=pending['commit_b'])
        receipt['remote_references'] = references
        checkout, storage = root / 'remote-verification', root / 'empty-lfs-storage'
        if not checkout.exists():
            checkout.mkdir()
            git('init', '-q', cwd=checkout)
            git('remote', 'add', 'origin', origin, cwd=checkout)
            git('config', '--local', 'lfs.storage', str(storage), cwd=checkout)
            if storage.exists() and any(storage.iterdir()):
                raise ValueError('the remote verification requires an initially empty independent LFS storage')
            git('fetch', '--depth=1', 'origin', f'refs/tags/{tag}:refs/tags/{tag}', cwd=checkout, env={'GIT_LFS_SKIP_SMUDGE': '1'})
            git('checkout', '--detach', pending['commit_b'], cwd=checkout, env={'GIT_LFS_SKIP_SMUDGE': '1'})
        if git('rev-parse', 'HEAD', cwd=checkout) != pending['commit_b']:
            raise ValueError('clean verification clone has another identity')
        include = ','.join(row['path'] for row in rows)
        if rows:
            git('lfs', 'fetch', f'--include={include}', '--exclude=', 'origin', tag, cwd=checkout)
            git('lfs', 'checkout', '--', *[row['path'] for row in rows], cwd=checkout)
        for row in rows:
            path = checkout / row['path']
            if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
                raise ValueError('downloaded entity is a pointer or has different bytes: '+row['path'])
            receipt['assets'].append(dict(path=row['path'], bytes=row['bytes'], sha256=row['sha256'], status='DOWNLOADED_AND_VERIFIED'))
        receipt.update(GIT_LFS='PUSH_VERIFIED', PUBLICATION='GIT_LFS_VERIFIED',
            independent_empty_LFS_storage=True, final_entities_only=True,
            tag_object_is_annotated=True, peeled_tag_equals_commit_b=True)
    except (subprocess.CalledProcessError, ValueError, OSError) as error:
        receipt.update(PUBLICATION='INCOMPLETE', error=public_record(str(error), assets),
            error_output=public_record(getattr(error, 'output', None), assets))
        write_json_atomic(receipt_path, receipt)
        return receipt
    if not shutil.which('gh') or subprocess.run(['gh', 'auth', 'status'], capture_output=True).returncode:
        receipt['RELEASE'] = 'UNAVAILABLE_API_CREDENTIALS'
        write_json_atomic(receipt_path, receipt)
        return receipt
    try:
        existing = subprocess.run(['gh', 'release', 'view', tag, '--repo', 'Orangekostar/Persist4D', '--json', 'url'], capture_output=True, text=True)
        if existing.returncode:
            notes = root / 'RELEASE_NOTES.md'
            notes.write_text(f'OGR-V1: {status["EXPERIMENT"]}; actual nonformal probes and diagnostic evidence. '
                             f'Frozen A: {pending["commit_a"]}; handoff B: {pending["commit_b"]}. See REPORT and FINAL_REVIEW.')
            subprocess.check_call(['gh', 'release', 'create', tag, '--repo', 'Orangekostar/Persist4D', '--verify-tag',
                '--title', 'Persist4D OGR-V1 measured evidence', '--notes-file', str(notes)])
        # This fact is already true before attachment. Keep the uploaded and
        # local receipt byte-identical rather than changing it after upload.
        receipt['RELEASE'] = 'CREATED_OR_UPDATED'
        write_json_atomic(receipt_path, receipt)
        subprocess.check_call(['gh', 'release', 'upload', tag, str(receipt_path), '--repo', 'Orangekostar/Persist4D', '--clobber'])
    except (OSError, subprocess.CalledProcessError) as error:
        receipt.update(RELEASE='API_OR_UPLOAD_FAILED', release_error=public_record(str(error), assets))
        write_json_atomic(receipt_path, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('publish',))
    parser.add_argument('--config', default='conf/observation_grounded_v1/default.yaml')
    args = parser.parse_args(argv)
    import yaml
    result = publish(local_assets(), yaml.safe_load((PROJECT / args.config).read_text()))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result.get('GIT_LFS') == 'PUSH_VERIFIED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
