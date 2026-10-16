"""Cached pooling must preserve official validation, order and exact values."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


def cache_type():
    name = 'scripts.revisit_prompt_pool_cache'
    assert importlib.util.find_spec(name) is not None, 'Exact official pool cache is missing'
    return importlib.import_module(name).OfficialPoolCache


@pytest.fixture
def population(tmp_path):
    from scripts.p6a_metrics import OfficialMetricAccumulator

    original = Path(__file__).resolve().parents[1] / 'artifacts/native_long_postselection_v1/deployment/rio.yaml'
    taxonomy = tmp_path / 'taxonomy.yaml'
    taxonomy.write_bytes(original.read_bytes())
    states = []
    for score in (.5, .9):
        masks = torch.ones(1, 300, dtype=torch.bool)
        target = dict(ids=torch.tensor([42]), labels=torch.tensor([5]), masks=masks,
                      changes=torch.tensor([0]), temporal_stages=torch.zeros(300, dtype=torch.long))
        prediction = dict(pred_masks=masks.T, pred_scores=torch.tensor([score]), pred_classes=torch.tensor([5]))
        metric = OfficialMetricAccumulator(mode='raw_local', dataset_spec=taxonomy)
        metric.update(prediction, target)
        states.append(metric.export_evidence())
    return taxonomy, states


def make_cache(tmp_path, *, pool=None, read_only=False, sources=None, runtime=None):
    from scripts.evaluate_revisit_prompt import pool_official_states

    return cache_type()(tmp_path / 'cache', pool=pool or pool_official_states,
                        source_paths=sources or {}, runtime_identity=runtime or {'test_runtime': '1'},
                        read_only=read_only)


def test_identical_inputs_reuse_exact_metrics_and_return_independent_copies(tmp_path, population):
    taxonomy, states = population
    cache = make_cache(tmp_path)
    expected = {f'raw_local_{key}': 1. for key in ('AP', 'AP50', 'AP25', 'REC', 'REC50', 'REC25')}
    first = cache(states, dataset_spec=taxonomy)
    assert first == expected
    first['raw_local_AP'] = -1.
    assert cache(copy.deepcopy(states), dataset_spec=taxonomy) == expected
    assert (cache.hits, cache.misses) == (1, 1)


def test_mutated_raw_state_with_unchanged_claimed_hash_is_not_a_hit(tmp_path, population):
    taxonomy, states = population
    cache = make_cache(tmp_path)
    cache(states, dataset_spec=taxonomy)
    states[0]['updates'] += 1
    with pytest.raises(ValueError, match='byte identity differs'):
        cache(states, dataset_spec=taxonomy)
    assert cache.hits == 0
    assert len(list((tmp_path / 'cache').glob('*.json'))) == 1


def test_reference_order_is_part_of_the_cache_identity(tmp_path, population):
    taxonomy, states = population
    cache = make_cache(tmp_path)
    cache(states, dataset_spec=taxonomy)
    cache(list(reversed(states)), dataset_spec=taxonomy)
    assert (cache.hits, cache.misses) == (0, 2)
    assert len(list((tmp_path / 'cache').glob('*.json'))) == 2


def test_taxonomy_content_change_cannot_reuse_an_old_value(tmp_path, population):
    taxonomy, states = population
    cache = make_cache(tmp_path)
    expected = cache(states, dataset_spec=taxonomy)
    taxonomy.write_text(taxonomy.read_text() + '\n# a changed taxonomy byte identity\n')
    assert cache(states, dataset_spec=taxonomy) == expected
    assert (cache.hits, cache.misses) == (0, 2)


def test_source_changes_are_rejected_and_new_runtime_uses_a_new_identity(tmp_path, population):
    taxonomy, states = population
    source = tmp_path / 'source.py'
    source.write_text('version = 1\n')
    cache = make_cache(tmp_path, sources={'extra_source': source})
    expected = cache(states, dataset_spec=taxonomy)
    another = make_cache(tmp_path, sources={'extra_source': source}, runtime={'test_runtime': '2'})
    assert another(states, dataset_spec=taxonomy) == expected
    assert another.hits == 0
    source.write_text('version = 2\n')
    with pytest.raises(RuntimeError, match='source identity changed'):
        cache(states, dataset_spec=taxonomy)


@pytest.mark.parametrize('damage', ['metric', 'key', 'malformed'])
def test_corrupt_cache_is_rejected(tmp_path, population, damage):
    taxonomy, states = population
    cache = make_cache(tmp_path)
    cache(states, dataset_spec=taxonomy)
    path, = (tmp_path / 'cache').glob('*.json')
    payload = json.loads(path.read_text())
    if damage == 'malformed':
        path.write_text('{')
    else:
        if damage == 'metric':
            payload['metrics']['raw_local_AP'] = -1.
        else:
            payload['key'] = '0' * 64
        path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='cache integrity'):
        cache(states, dataset_spec=taxonomy)


def test_read_only_consumer_reuses_entries_and_missing_entries_compute_without_writes(tmp_path, population):
    taxonomy, states = population
    writer = make_cache(tmp_path)
    expected = writer(states, dataset_spec=taxonomy)
    before = {p.name: p.read_bytes() for p in (tmp_path / 'cache').iterdir()}
    reader = make_cache(tmp_path, read_only=True)
    assert reader(states, dataset_spec=taxonomy) == expected
    assert reader(list(reversed(states)), dataset_spec=taxonomy) == expected
    assert (reader.hits, reader.misses) == (1, 1)
    assert {p.name: p.read_bytes() for p in (tmp_path / 'cache').iterdir()} == before


def test_undefined_official_metrics_remain_none_and_failures_are_not_cached(tmp_path, population):
    from scripts.p6a_metrics import OfficialMetricAccumulator

    taxonomy, _ = population
    metric = OfficialMetricAccumulator(mode='raw_local', dataset_spec=taxonomy)
    metric.update(dict(pred_masks=torch.zeros(300, 0, dtype=torch.bool),
                       pred_scores=torch.empty(0), pred_classes=torch.empty(0, dtype=torch.long)),
                  dict(ids=torch.empty(0, dtype=torch.long), labels=torch.empty(0, dtype=torch.long),
                       masks=torch.zeros(0, 300, dtype=torch.bool), changes=torch.empty(0, dtype=torch.long),
                       temporal_stages=torch.zeros(300, dtype=torch.long)))
    cache = make_cache(tmp_path)
    states = [metric.export_evidence()]
    assert set(cache(states, dataset_spec=taxonomy).values()) == {None}
    assert set(cache(states, dataset_spec=taxonomy).values()) == {None}
    with pytest.raises(ValueError, match='nonempty'):
        cache([], dataset_spec=taxonomy)
    assert len(list((tmp_path / 'cache').glob('*.json'))) == 1
