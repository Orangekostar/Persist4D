"""Same-class diagnostics use distinct, reliable current GT and saved matches."""
import importlib
import importlib.util
import json

import pytest


def counts(*args, **kwargs):
    name = 'scripts.revisit_prompt_cooccurrence'
    assert importlib.util.find_spec(name) is not None, 'Same-class diagnostic helper is missing'
    return importlib.import_module(name).cooccurrence_counts(*args, **kwargs)


def test_counts_distinct_same_class_entities_and_preserves_recoveries_and_losses():
    result = counts([1, 2, 3, 4, -1, 5], [5, 5, 7, 7, 5, 255],
                    {'M0': [1], 'M3': [1, 3], 'M4': [2, 3]},
                    ambiguities=[[4, 40]], valid_class_ids=[5, 7])
    assert result['current_gt'] == 6
    assert result['excluded_identity_gt'] == 2
    assert result['excluded_class_gt'] == 1
    assert result['eligible_gt'] == 3
    assert result['cooccurring_gt'] == 2
    assert result['same_class_pairs'] == 1
    assert result['matched_by_variant'] == {'M0': 1, 'M3': 1, 'M4': 1}
    assert result['recall_by_variant'] == {'M0': .5, 'M3': .5, 'M4': .5}
    assert result['paired']['M4_vs_M0'] == {'recovered': 1, 'lost': 1, 'net': 0}
    assert result['paired']['M4_vs_M3'] == {'recovered': 1, 'lost': 1, 'net': 0}


def test_singletons_and_empty_populations_have_undefined_subset_recall():
    for identities, labels in (([], []), ([1, 2], [5, 7])):
        result = counts(identities, labels, {'M0': [], 'M3': [], 'M4': []},
                        ambiguities=None, valid_class_ids=[5, 7])
        assert result['same_class_pairs'] == result['cooccurring_gt'] == 0
        assert result['recall_by_variant'] == {'M0': None, 'M3': None, 'M4': None}


def test_each_entity_is_counted_once_but_all_unordered_same_class_pairs_are_counted():
    result = counts([1, 2, 3], [5, 5, 5], {'M0': [], 'M3': [1], 'M4': [1, 2]},
                    ambiguities=[], valid_class_ids=[5])
    assert result['cooccurring_gt'] == 3 and result['same_class_pairs'] == 3
    assert result['matched_by_variant'] == {'M0': 0, 'M3': 1, 'M4': 2}
    assert result['recall_by_variant']['M4'] == 2 / 3
    assert result['paired']['M4_vs_M0'] == {'recovered': 2, 'lost': 0, 'net': 2}


@pytest.mark.parametrize('ids,labels,matches', [
    ([1, 1], [5, 7], {'M0': [], 'M3': [], 'M4': []}),
    ([1], [], {'M0': [], 'M3': [], 'M4': []}),
    ([1], [5], {'M0': [99], 'M3': [], 'M4': []}),
])
def test_misaligned_truth_or_match_population_is_rejected(ids, labels, matches):
    with pytest.raises(ValueError, match='population'):
        counts(ids, labels, matches, ambiguities=[], valid_class_ids=[5, 7])


def test_default_cli_does_not_load_data_or_write_results(capsys):
    name = 'scripts.revisit_prompt_cooccurrence'
    assert importlib.util.find_spec(name) is not None, 'Same-class diagnostic helper is missing'
    module = importlib.import_module(name)
    assert module.main([]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'DRY_RUN_NOT_STARTED'
