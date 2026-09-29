"""Selection is CAL-only, complete-population, multi-module and tolerance-aware."""

import pytest
import torch

from scripts.short_module_evaluation import (
    build_cal_lock,
    build_shortlist,
    classify_gain,
)


def test_all_six_cal_locks_require_three_complete_trained_points_and_ignore_zero():
    rows = [{"module": arm, "seed": 45, "step": step, "status": "COMPLETE",
             "metrics": {"T2": .2 if step else .99, "T1": .3}}
            for arm in ("Q1", "Q2", "Q3", "M0", "M1", "M2")
            for step in (0, 500, 1000, 1500)]
    lock = build_cal_lock(rows)
    assert set(lock["selected_updates"].values()) == {500}
    with pytest.raises(ValueError):
        build_cal_lock(rows[:-1])
    rows[-1]["status"] = "PARTIAL"
    with pytest.raises(ValueError):
        build_cal_lock(rows)


def test_gain_thresholds_and_multiple_independent_shortlist():
    assert classify_gain(-.001, .1, True) == "NEGATIVE"
    assert classify_gain(.0000005, 0, True) == "NO_GAIN"
    assert classify_gain(.01, -.003, True) == "T2_ONLY_TRADEOFF"
    assert classify_gain(.01, 0, False) == "T2_ONLY_TRADEOFF"
    assert classify_gain(.003, -.002, True) == "MODEST_GAIN"
    assert classify_gain(.005, -.002, True) == "TARGET_GAIN"
    rows = [{"module": arm, "classification": "TARGET_GAIN",
             "evidence_level": "DEVELOPMENT_REPLICATED"} for arm in ("Q1", "Q2", "M1")]
    shortlist = build_shortlist(rows)
    assert [r["module"] for r in shortlist["confirmed"]] == ["Q1", "Q2", "M1"]
    assert shortlist["slot_policy"] == "ALTERNATIVES_NOT_STACKABLE"
    assert build_shortlist([])["confirmed"] == []


def test_sel_rejects_unlocked_checkpoint_before_reading_predictions(tmp_path, monkeypatch):
    from scripts import short_module_evaluation as evaluation
    from scripts.short_module_screen import write_json

    monkeypatch.setattr(evaluation, "ARTIFACTS", tmp_path)
    write_json(tmp_path / "selection/CAL_LOCK.json", {
        "selected_updates": {arm: 500 for arm in evaluation.ARMS}})
    with pytest.raises(ValueError, match="fixed CAL point"):
        evaluation.evaluate_point(tmp_path, module="Q1", step=1000, seed=45, role="SEL", device="cpu")


def test_checkpoint_rejects_wrong_parent_and_data_identity(tmp_path, monkeypatch):
    from scripts import short_module_evaluation as evaluation
    from scripts.short_module_screen import write_json

    monkeypatch.setattr(evaluation, "ARTIFACTS", tmp_path / "public")
    write_json(evaluation.ARTIFACTS / "RUN_CONFIG.json", {
        "protocol": {"parent": {"sha256": "parent"}}, "parent_dimensions": {"feature_dim": 2}})
    write_json(tmp_path / "EXPORT_INDEX.json", {"cache_identity_sha256": "data"})
    path = tmp_path / "training/Q1/seed45/update=0500.pt"
    path.parent.mkdir(parents=True)
    meta = {"module": "Q1", "seed": 45, "optimizer_updates": 500,
            "base_sha256": "wrong", "dimensions": {"feature_dim": 2}, "cache_identity_sha256": "wrong"}
    torch.save({"metadata": meta}, path)
    with pytest.raises(ValueError, match="parent"):
        evaluation.load_head(tmp_path, "Q1", 500, 45, "cpu")
    meta["base_sha256"] = "parent"
    torch.save({"metadata": meta}, path)
    with pytest.raises(ValueError, match="input/cache"):
        evaluation.load_head(tmp_path, "Q1", 500, 45, "cpu")


def test_incomplete_population_returns_null_without_reusing_metric_state(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import short_module_evaluation as evaluation
    from scripts.short_module_screen import write_json

    public = tmp_path / "public"
    monkeypatch.setattr(evaluation, "ARTIFACTS", public)
    records = [{"role": "CAL", "horizon": h, "input_id": str(h), "reference_id": "reference",
                "logical_unit_id": str(h)} for h in (1, 2)]
    write_json(public / "SHORT_POPULATION.json", {"records": records})
    write_json(tmp_path / "EXPORT_INDEX.json", {"status": "COMPLETE", "cache": str(tmp_path),
        "cache_identity_sha256": "fixture", "entries": [{"input_id": "1", "prediction": {"file": "prediction.pt"},
                                                        "targets": {"file": "target.pt"}}]})
    spec = tmp_path / "dataset.yaml"
    spec.write_text("name: fixture\n")
    write_json(tmp_path / "assets.local.json", {"metric_dataset_spec": str(spec)})
    torch.save({}, tmp_path / "prediction.pt")
    torch.save({"target": {"masks": torch.ones(1, 2, dtype=torch.bool)}}, tmp_path / "target.pt")
    parent = SimpleNamespace(pred_masks=torch.ones(2, 1, dtype=torch.bool), pred_scores=torch.ones(1),
                             pred_classes=torch.ones(1, dtype=torch.long), source_query_ids=torch.zeros(1),
                             source_class_ids=torch.zeros(1))
    monkeypatch.setattr(evaluation, "load_head", lambda *args: (None, None))
    monkeypatch.setattr(evaluation, "materialization_system", lambda *args: None)
    monkeypatch.setattr(evaluation, "unpack_prediction", lambda saved: parent)
    monkeypatch.setattr(evaluation, "unpack_bool_matrix", lambda value: value)
    monkeypatch.setattr(evaluation, "apply_module", lambda *args, **kwargs: vars(parent))
    torch.save({"parent": {}, "descriptor": {}}, tmp_path / "prediction.pt")
    accumulators = []

    class Accumulator:
        def __init__(self, **kwargs):
            self._updates = 0
            accumulators.append(self)

        def update(self, prediction, target):
            self._updates += 1

        def compute(self):
            return {"raw_local_AP": .25, "online_t-mAP": .5}

        def export_evidence(self):
            return {"updates": self._updates}

    monkeypatch.setattr(evaluation, "OfficialMetricAccumulator", Accumulator)
    result = evaluation.evaluate_point(tmp_path, module="B0", step=0, seed=45, role="CAL", device="cpu")
    assert result["status"] == "PARTIAL"
    assert result["expected"] == {1: 1, 2: 1}
    assert result["completed"] == {1: 1, 2: 0}
    assert result["metrics"] == {"T1": .25, "T2": None}
    assert len(accumulators) == len({id(a) for a in accumulators}) == 5
    assert [a._updates for a in accumulators] == [1, 0, 0, 0, 1]


def test_screen_failure_keeps_other_five_arms_and_full_expected_population(tmp_path, monkeypatch):
    from scripts import short_module_evaluation as evaluation
    from scripts.short_module_screen import read_json, write_json

    monkeypatch.setattr(evaluation, "ARTIFACTS", tmp_path)
    write_json(tmp_path / "selection/CAL_LOCK.json", {"selected_updates": {a: 500 for a in evaluation.ARMS}})
    baseline = {"status": "COMPLETE", "metrics": {"T1": .4, "T2": .3}, "expected": {"1": 23, "2": 24},
                "by_reference": [{"reference_id": "ref", "H": 2, "AP": .3}]}
    write_json(tmp_path / "BASELINE.json", {"SEL": baseline})
    called = []

    def point(root, **kwargs):
        called.append(kwargs["module"])
        if kwargs["module"] == "M0":
            raise OSError("unreadable cache")
        return {**baseline, "module": kwargs["module"], "seed": 45, "step": 500}

    monkeypatch.setattr(evaluation, "evaluate_point", point)
    rows = evaluation.screen(tmp_path, device="cpu")
    assert set(called) == set(evaluation.ARMS)
    assert len(rows) == 6
    failed = next(r for r in rows if r["module"] == "M0")
    assert failed["classification"] == "BLOCKED"
    assert failed["SEL_T2_AP"] is None
    assert read_json(tmp_path / "selection/SCREEN_RESULTS.json")["M0"]["expected"] == {"1": 23, "2": 24}
