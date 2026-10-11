import hashlib

import pytest

from scripts.native_long_evidence_package import official_bytes, validate_state


def test_metric_package_rejects_extra_point_payload_and_modified_statistics():
    state = {'schema_version': 1, 'mode': 'strict_online', 'states': [], 'updates': 0}
    state['sha256'] = hashlib.sha256(official_bytes(state)).hexdigest()
    validate_state(state)
    with pytest.raises(ValueError, match='sufficient-state-only'):
        validate_state({**state, 'point_masks': [[True]]})
    with pytest.raises(ValueError, match='digest'):
        validate_state({**state, 'updates': 1})
