import copy
import hashlib

import pytest

from scripts.native_long_delivery import portable_config, validate_cal_selection


def test_portable_config_relocates_both_encoder_references_and_taxonomy(tmp_path):
    encoder, taxonomy = tmp_path / 'encoder.pth', tmp_path / 'rio.yaml'
    encoder.write_bytes(b'bound encoder')
    taxonomy.write_bytes(b'bound taxonomy')
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    payload = {'metadata': {'pretrained_encoder_sha256': digest(encoder)}, 'config': {
        'backbone': {'name': '/old/encoder.pth'},
        'model': {'config': {'backbone': {'name': '/old/encoder.pth'}}},
        'instance_metric': {'dataset': '/old/rio.yaml'}, 'native_long': {'arm': 'E3'}}}
    original = copy.deepcopy(payload)
    config = portable_config(payload, encoder, taxonomy, digest(taxonomy))
    assert config.backbone.name == config.model.config.backbone.name == str(encoder.resolve())
    assert config.instance_metric.dataset == str(taxonomy.resolve())
    assert config.native_long.arm == 'E3'
    assert payload == original
    with pytest.raises(ValueError, match='taxonomy'):
        portable_config(payload, encoder, taxonomy, '0' * 64)
    encoder.write_bytes(b'different encoder')
    with pytest.raises(ValueError, match='encoder'):
        portable_config(payload, encoder, taxonomy, digest(taxonomy))


def test_cal_selection_requires_stopped_trajectories_seed_and_population():
    lock = {'schema': 1, 'seed': 45, 'all_planned_trajectories_stopped_before_SEL': True,
            'population_sha256': 'abc', 'selected_updates': {'E2': 29700}}
    validate_cal_selection(lock, 'E2', 29700, 'abc')
    for changed in ({'seed': 46}, {'population_sha256': 'other'},
                    {'all_planned_trajectories_stopped_before_SEL': False},
                    {'selected_updates': {'E2': 23760}}):
        with pytest.raises(ValueError, match='CAL'):
            validate_cal_selection({**lock, **changed}, 'E2', 29700, 'abc')
