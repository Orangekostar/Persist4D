from datasets.preprocessing.RScan_preprocessing import RScanPreprocessing


def test_official_builder_never_repeats_scan_to_fake_long_horizon() -> None:
    builder = object.__new__(RScanPreprocessing)
    builder.scenes = {
        "a": {"scan_id": 0, "rescan_id": 0},
        "b": {"scan_id": 0, "rescan_id": 1},
    }

    assert builder.create_sequences("sliding", sequence_length=3, seed=45) == []
