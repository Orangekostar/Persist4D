"""A predecessor lock must not bypass measured fresh numerical admission."""
import json

import pytest

from scripts import continue_revisit_prompt_formal as controller
from scripts.revisit_prompt_assets import numeric_sources


def setup_completed_run(tmp_path, monkeypatch, values):
    sources = numeric_sources()
    counts = {str(h): 1 for h in range(1, 6)}
    assets = dict(run_root=str(tmp_path), python="unused", hashes=dict(parent="parent"))
    public = tmp_path / "public"
    public.mkdir()
    (public / "DATA_MANIFEST.json").write_text(json.dumps(dict(counts=dict(CAL=counts))))
    monkeypatch.setattr(controller, "ARTIFACTS", public)
    monkeypatch.setattr(controller, "local_assets", lambda: assets)
    monkeypatch.setattr(controller, "read_run_plan", lambda _: dict(N=400))
    monkeypatch.setattr(controller, "ensure_job", lambda *_, **__: None)
    monkeypatch.setattr(controller, "evaluate_endpoint", lambda *_: None)
    calls = []
    monkeypatch.setattr(controller, "run_campaign", lambda *_, **__: calls.append("dispatch") or dict(
        experiment_status="COMPLETE", scientific_status="INCONCLUSIVE"))
    paths = []
    for repeat, value in enumerate(values):
        folder = tmp_path / "evaluation/M0-s146-u000000/CAL" / f"repeat{repeat}"
        folder.mkdir(parents=True)
        request = dict(variant="M0", adapter_seed=146, step=0, role="CAL", repeat_index=repeat,
                       parent_sha256="parent", numeric_sources=sources, expected_inputs=counts, input_count=5)
        status = dict(request, status="COMPLETE", completed_inputs=counts)
        result = dict(status="COMPLETE", request=request, rows=[dict(
            horizon=h, AP=value if h == 5 else .4, expected_inputs=1, completed_inputs=1)
            for h in range(1, 6)])
        for filename, payload in (("STATUS.json", status), ("RESULTS.json", result)):
            p = folder / filename
            p.write_text(json.dumps(payload))
            paths.append(p)
    return assets, calls, paths


def test_fresh_variance_blocks_dispatch_and_preserves_original_results(tmp_path, monkeypatch):
    _, calls, paths = setup_completed_run(tmp_path, monkeypatch, [.135, .135, .164])
    before = {p:p.read_bytes() for p in paths}
    assert controller.main() == 2
    assert calls == []
    status = json.loads((tmp_path / "CONTROLLER_STATUS.json").read_text())
    assert status["status"] == "BLOCKED_NUMERICAL_VARIANCE"
    assert status["fresh_m0_admission"]["epsilon_run_pp"] == pytest.approx(2.9)
    assert {p:p.read_bytes() for p in paths} == before


def test_fresh_stable_repeats_allow_campaign_dispatch(tmp_path, monkeypatch):
    _, calls, _ = setup_completed_run(tmp_path, monkeypatch, [.5, .5, .502])
    assert controller.main() == 0
    assert calls == ["dispatch"]
    admission = json.loads((tmp_path / "FRESH_M0_ADMISSION.json").read_text())
    assert admission["status"] == "ADMITTED"
    assert admission["epsilon_run_pp"] == pytest.approx(.2)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_undefined_or_nonfinite_fresh_ap_never_admits(tmp_path, monkeypatch, value):
    _, calls, _ = setup_completed_run(tmp_path, monkeypatch, [.5, .5, value])
    with pytest.raises(ValueError, match="undefined or nonfinite"):
        controller.main()
    assert calls == []


def test_incomplete_fresh_population_never_admits(tmp_path, monkeypatch):
    _, calls, paths = setup_completed_run(tmp_path, monkeypatch, [.5, .5, .5])
    status_path = paths[-2]
    status = json.loads(status_path.read_text())
    status["completed_inputs"]["5"] = 0
    status_path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="complete frozen CAL population"):
        controller.main()
    assert calls == []


def test_policy_admission_preserves_original_failed_report(tmp_path, monkeypatch):
    assets, _, _ = setup_completed_run(tmp_path, monkeypatch, [.5,.5,.5])
    old = tmp_path / "FRESH_M0_ADMISSION.json"
    old.write_text('{"status":"BLOCKED_NUMERICAL_VARIANCE","epsilon_run_pp":2.89}')
    before = old.read_bytes()
    admission = controller.fresh_m0_admission(assets, report_name="EVALUATION_M0_ADMISSION.json")
    assert admission["status"] == "ADMITTED"
    assert old.read_bytes() == before
    assert (tmp_path / "EVALUATION_M0_ADMISSION.json").exists()
