from scripts.short_module_results import replication_requests


def test_replication_adds_direct_controls_at_fixed_steps_without_new_epoch_search():
    rows = [{"module": "Q2", "classification": "MODEST_GAIN", "selected_step": 500},
            {"module": "M2", "classification": "TARGET_GAIN", "selected_step": 1000},
            {"module": "Q1", "classification": "TARGET_GAIN", "selected_step": 1500},
            {"module": "M1", "classification": "NEGATIVE", "selected_step": 500}]
    assert replication_requests(rows) == {"Q1": [500, 1500], "Q2": [500],
                                          "M1": [1000], "M2": [1000]}
    assert replication_requests([]) == {}
