import pytest


def test_bridge_context_is_causal_and_prefix_matches_pair_only_at_first_two_visits():
    from scripts.native_long_memory_bridge import context_record

    unit = {'source_unit': 'master:canonical', 'scan_ids': list('abcde'),
            'physical_scan_ids': list('ABCDE'), 'scan_indices': list(range(5)),
            'horizon': 5, 'input_id': 'original', 'reference_id': 'r', 'domain': 'rio'}
    for h in range(1, 6):
        full, local = (context_record(unit, h, mode) for mode in ('full', 'local'))
        assert full['scan_ids'] == list('abcde')[:h]
        assert local['scan_ids'] == list('abcde')[:h][-2:]
        assert (full['input_id'] == local['input_id']) is (h <= 2)
        assert full['draw_seed_key'] == local['draw_seed_key']
    with pytest.raises(ValueError):
        context_record(unit, 6, 'full')


def test_identity_pooling_uses_counts_and_preserves_missing_denominators():
    from scripts.native_long_memory_bridge import pool_identity

    a = {'deployment_id_switches': 1, 'identity_transition_opportunities': 2,
         'fragmentation_count': 0, 'fragmentation_opportunities': 0,
         'merge_count': 0, 'merge_opportunities': 0, 'gap_opportunities': 1,
         'recovery_attempts': 1, 'correct_recoveries': 1}
    b = {**a, 'deployment_id_switches': 0, 'identity_transition_opportunities': 8,
         'gap_opportunities': 9, 'recovery_attempts': 0, 'correct_recoveries': 0}
    result = pool_identity([a, b])
    assert result['normalized_id_switch_rate'] == .1
    assert result['gap_recovery_recall'] == .1
    assert result['gap_recovery_accuracy'] == 1.
    assert result['fragmentation_rate'] is None


def test_current_target_sorts_ids_and_preserves_empty_current_visit():
    import torch

    from scripts.native_long_memory_bridge import current_target

    target = {'ids': torch.tensor([8, 3]), 'labels': torch.tensor([2, 1]),
              'temporal_stages': torch.tensor([0, 1, 1, 2]),
              'masks': torch.tensor([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=torch.bool)}
    current = current_target(target, 2)
    assert current['ids'].tolist() == [3, 8]
    assert current['labels'].tolist() == [1, 2]
    assert current['masks'].tolist() == [[True, False], [False, True]]
    empty = current_target(target, 3)
    assert empty['ids'].shape == (0,)
    assert empty['labels'].shape == (0,)
    assert empty['masks'].shape == (0, 1)


def test_common_truth_audits_window_label_drift_but_rejects_point_misalignment():
    import torch

    from scripts.native_long_memory_bridge import audit_current_alignment

    full = {'ids': torch.tensor([2]), 'labels': torch.tensor([17]),
            'masks': torch.tensor([[True, False]])}
    local = {**full, 'labels': torch.tensor([0])}
    assert audit_current_alignment(full, local) == [{'id': 2, 'full_label': 17, 'local_label': 0}]
    assert local['labels'].tolist() == [0]
    with pytest.raises(ValueError, match='point-order'):
        audit_current_alignment(full, {**local, 'masks': torch.tensor([[False, True]])})
    with pytest.raises(ValueError, match='point-order'):
        audit_current_alignment(full, {**local, 'ids': torch.tensor([3])})


def test_common_truth_reconstructs_filtered_instance_from_raw_point_ids():
    import torch

    from scripts.native_long_memory_bridge import project_common_truth

    full = {'ids': torch.tensor([3, 46]), 'labels': torch.tensor([5, 8]),
            'masks': torch.tensor([[True, False, False], [False, True, True]])}
    aligned = project_common_truth(full, torch.tensor([3, 46, 46]))
    assert torch.equal(aligned['masks'], full['masks'])
    with pytest.raises(ValueError, match='point-order'):
        project_common_truth(full, torch.tensor([46, 3, 46]))


@pytest.mark.parametrize('empty', [False, True])
def test_bridge_payload_runs_frozen_trackers_and_official_trajectory(tmp_path, empty):
    import dataclasses
    from pathlib import Path

    import torch
    import yaml

    from scripts.evaluate_persist4d_p6a import (
        build_temporal_target,
        build_tracker_factories,
        cache_payload_to_frozen_observation,
    )
    from scripts.native_long_budget import write_json_atomic
    from scripts.native_long_memory_bridge import build_local_payload, pool_identity
    from scripts.p6a_cache import SCHEMA_VERSION, validate_cache_payload
    from scripts.rescene_task_postprocess import OfficialTaskPrediction
    from scripts.system_comparison_metrics import (
        compute_deployment_identity_metrics,
        match_identity_update,
    )
    from scripts.system_comparison_v2_cache import (
        build_task_sidecar,
        task_sidecar_digest,
    )
    from scripts.system_comparison_v2_inference import (
        OfficialCandidateTrajectoryAccumulator,
    )

    unit = {'source_unit': 'master:canonical', 'reference_id': 'reference',
            'scan_ids': [f'scene0000_0{i}' for i in range(5)]}
    provenance = {'source_commit': '1' * 40, 'checkpoint_sha256': '2' * 64,
                  'config_sha256': '3' * 64, 'dataset_sha256': '4' * 64}
    settings = yaml.safe_load((Path(__file__).resolve().parents[1] / 'conf/p6a/default.yaml').read_text())
    factories = build_tracker_factories(settings)
    trackers = {m: factories[m](unit['source_unit']) for m in ('B2', 'B4')}
    trajectories = {m: OfficialCandidateTrajectoryAccumulator(score_reducer='mean') for m in trackers}
    updates = {m: [] for m in trackers}
    payloads = []
    for h in range(1, 6):
        visible = not empty and h != 3
        masks = torch.tensor([[visible, visible, False]], dtype=torch.bool)
        probabilities = torch.zeros(1, 19)
        probabilities[0, 0], probabilities[0, 18] = .9, .1
        observation = {'features': torch.nn.functional.one_hot(torch.tensor([0]), 128).float(),
                       'class_prob': probabilities, 'confidence': torch.tensor([.9]),
                       'valid': torch.tensor([visible]), 'masks': masks,
                       'mask_support': masks.sum(1), 'local_query_ids': torch.tensor([0])}
        current = {'ids': torch.tensor([10] if visible else [], dtype=torch.long),
                   'labels': torch.zeros(int(visible), dtype=torch.long), 'masks': masks[:int(visible)]}
        payload = build_local_payload(unit, h, observation, current, provenance)
        validate_cache_payload(payload)
        assert payload['schema_version'] == SCHEMA_VERSION
        assert payload['key']['stage_index'] == h - 1
        assert payload['key']['local_window_scan_ids'] == unit['scan_ids'][:h][-2:]
        payloads.append(payload)
        prediction = OfficialTaskPrediction(
            pred_masks=masks.T, pred_scores=torch.tensor([.9]), pred_classes=torch.tensor([0]),
            source_query_ids=torch.tensor([0]), source_class_ids=torch.tensor([0]),
            temporal_stages=torch.full((3,), min(h, 2) - 1, dtype=torch.long),
            latest_stage_index=min(h, 2) - 1, latest_stage_masks=masks.T)
        sidecar = build_task_sidecar(raw_cache_payload=payload, official_prediction=prediction,
                                     protocol_manifest_sha256='5' * 64)
        fingerprint = task_sidecar_digest(sidecar)
        for method, tracker in trackers.items():
            step = tracker.step(cache_payload_to_frozen_observation(payload), stage_id=h - 1)
            queries = [q for q, value in enumerate(step.track_ids) if value is not None]
            updates[method].append(match_identity_update(
                horizon=h, gt_ids=current['ids'], gt_classes=current['labels'], gt_masks=current['masks'],
                issued_ids=torch.tensor([step.track_ids[q] for q in queries], dtype=torch.long),
                pred_classes=torch.zeros(len(queries), dtype=torch.long), pred_masks=masks[queries].T))
            trajectories[method].add_stage(sidecar, step)
            assert trajectories[method].snapshot().prediction['pred_masks'].shape[0] == 3 * h
        assert task_sidecar_digest(sidecar) == fingerprint
        target = build_temporal_target(payloads)
        assert target['temporal_stages'].tolist() == [s for s in range(h) for _ in range(3)]
    metrics = {m: compute_deployment_identity_metrics(rows) for m, rows in updates.items()}
    if not empty:
        assert metrics['B4']['gap_opportunities'] == 1
        assert metrics['B4']['correct_recoveries'] == 1
        assert metrics['B2']['correct_recoveries'] == 0
    else:
        assert metrics['B4']['gap_opportunities'] == 0
    state = trackers['B4'].state
    assert int(state.occupied.sum()) == int(not empty)
    assert sum(t.numel() * t.element_size() for t in vars(state).values()
               if isinstance(t, torch.Tensor)) > 0
    write_json_atomic(tmp_path / 'evidence.json', {
        'identity_updates': {m: [dataclasses.asdict(x) for x in rows] for m, rows in updates.items()},
        'pooled': {m: pool_identity([row]) for m, row in metrics.items()}})
