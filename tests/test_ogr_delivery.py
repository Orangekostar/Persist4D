"""Cost locks and asset boundaries that prevent invalid GPU execution."""

import pytest


def test_budget_resume_never_renews_run_cap_or_double_deducts_reservations(tmp_path):
    from scripts.ogr_assets import BudgetJournal

    journal = BudgetJournal(tmp_path, prior=1760.6677541187635, external_reserved=1.)
    event = journal.reserve('integration', 'I_INTEGRATION_REPAIR', 1.5, gpu=0)
    assert journal.snapshot()['remaining_gpu_hours'] == pytest.approx(48.5)
    # Reopen represents an actual controller resume, not another 50-hour grant.
    resumed = BudgetJournal(tmp_path, prior=1760.6677541187635, external_reserved=1.)
    assert resumed.snapshot()['remaining_gpu_hours'] == pytest.approx(48.5)
    resumed.finish(event, seconds=1800., exit_code=1)
    assert resumed.snapshot()['remaining_gpu_hours'] == pytest.approx(49.5)
    assert resumed.snapshot()['spent_gpu_hours'] == .5
    with pytest.raises(ValueError):
        resumed.reserve('integration', 'I_INTEGRATION_REPAIR', .1, gpu=0)
    with pytest.raises(ValueError):
        resumed.reserve('too-large', 'I_INTEGRATION_REPAIR', 3.6, gpu=0)


def test_shared_lock_enforces_two_single_gpu_jobs_and_tight_lifetime_limit(tmp_path):
    from scripts.ogr_assets import BudgetJournal

    journal = BudgetJournal(tmp_path, prior=1810., external_reserved=1.)
    assert journal.snapshot()['initial_available_gpu_hours'] == 4.
    journal.reserve('a', 'D_DIAGNOSTIC', 1., gpu=0)
    with pytest.raises(ValueError):
        journal.reserve('same-gpu', 'T_FIRST', 1., gpu=0)
    journal.reserve('b', 'T_FIRST', 1., gpu=1)
    with pytest.raises(ValueError):
        journal.reserve('third', 'T_FIRST', .1, gpu=2)
    assert journal.snapshot()['remaining_gpu_hours'] == 2.


def test_gpu_worker_refuses_unaccounted_execution(monkeypatch):
    from scripts.ogr_campaign import train_worker

    monkeypatch.delenv('OGR_CHARGED_EVENT_ID', raising=False)
    with pytest.raises(ValueError, match='charged'):
        train_worker(None, None)


def test_public_budget_export_contains_both_peer_settlements(tmp_path):
    import json
    from scripts.ogr_assets import BudgetJournal

    journal = BudgetJournal(tmp_path / 'private', prior=1760.)
    a = journal.reserve('a', 'I_INTEGRATION_REPAIR', .5, gpu=0)
    b = journal.reserve('b', 'I_INTEGRATION_REPAIR', .5, gpu=1)
    journal.finish(a, seconds=360., exit_code=0)
    journal.finish(b, seconds=720., exit_code=1)
    exported = journal.export(tmp_path / 'public')
    assert exported['spent_gpu_hours'] == pytest.approx(.3)
    assert exported['active_jobs'] == []
    ledger = [json.loads(row) for row in (tmp_path / 'public/COST_LEDGER.jsonl').read_text().splitlines()]
    assert len(ledger) == 2 and {row['status'] for row in ledger} == {'COMPLETE', 'FAILED'}


def test_publication_checks_peeled_annotated_tag_not_tag_object():
    from scripts.ogr_delivery import verify_remote_references

    refs = {'refs/heads/research/test': 'B', 'refs/tags/test-v1': 'TAG_OBJECT', 'refs/tags/test-v1^{}': 'B'}
    assert verify_remote_references(refs, branch='research/test', tag='test-v1', commit_b='B')
    with pytest.raises(ValueError):
        verify_remote_references({**refs, 'refs/tags/test-v1^{}': 'OLD'}, branch='research/test', tag='test-v1', commit_b='B')
    with pytest.raises(ValueError):
        verify_remote_references({key: value for key, value in refs.items() if key != 'refs/tags/test-v1'},
                                 branch='research/test', tag='test-v1', commit_b='B')


def test_reference_interval_preserves_NA_without_redraw_and_seed_pairing():
    from scripts.ogr_statistics import conditional_interval, paired_seed_mean

    unavailable = conditional_interval([.1]*949+[None]*51)
    assert unavailable['valid_draws'] == 949 and unavailable['conditional_valid_draw_interval'] is None
    interval = conditional_interval([.1]*950+[None]*50)
    assert interval['NA_draws'] == 50 and interval['conditional_valid_draw_interval'] == [.1, .1]
    assert interval['redraw_invalid'] is False
    assert paired_seed_mean([.2, -.1]) == pytest.approx(.05)
    assert paired_seed_mean([.2, None]) is None
