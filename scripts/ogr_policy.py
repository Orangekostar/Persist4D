"""OGR-V1 pure-CPU budget/selection/queue reference.

This is not a training implementation. Copy or adapt into scripts/ogr_policy.py.
Upstream callers must additionally verify real input sets, source identities and
complete evaluator evidence. Run this file for bounded policy self-tests.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

VARIANTS = ('R-TARGET', 'R-ID', 'R-QFB', 'R-OBS')
CONTROLS = VARIANTS[:3]
TOL = 1e-6


def finite(value: float, name: str, low: float = 0.0, high: float = math.inf) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a real number')
    result = float(value)
    if not math.isfinite(result) or not low <= result <= high:
        raise ValueError(f'{name} out of range')
    return result


def new_budget(prior: float, *, lifetime_cap: float = 1823.0,
               active_reservations: float = 0.0, own_spent: float = 0.0,
               own_reservations: float = 0.0) -> float:
    prior = finite(prior, 'settled prior')
    cap = finite(lifetime_cap, 'lifetime cap')
    reserved = finite(active_reservations, 'active reservations')
    spent=finite(own_spent, 'this run settled')
    own_reserved=finite(own_reservations, 'this run active reservations')
    total=max(0.0, min(50.0, cap - prior - reserved - 8.0))
    return max(0.0, total-spent-own_reserved)


def select_n(timed: Mapping[str, tuple[float, float]], *, available: float,
             nontraining_forecast: float, evaluation_forecast: float) -> int | None:
    """timed maps each variant to (startup seconds, p90 update seconds).

    nontraining_forecast includes actual settled probe/diagnostic/repair cost,
    remaining non-training cost and evaluation_forecast; do not double count.
    """
    if set(timed) != set(VARIANTS):
        raise ValueError('all four variants must have measured costs')
    available = finite(available, 'available')
    other = finite(nontraining_forecast, 'nontraining forecast')
    evaluation = finite(evaluation_forecast, 'evaluation forecast')
    if other < evaluation:
        raise ValueError('nontraining forecast must include evaluation')
    if evaluation > 12.0:
        return None
    for variant, values in timed.items():
        if len(values) != 2:
            raise ValueError('cost row must contain startup and update seconds')
        finite(values[0], variant+' startup')
        if finite(values[1], variant+' update') <= 0:
            raise ValueError('update duration must be positive')
    for n in (1200, 800, 400):
        cost = {v: 1.25*(s+n*t)/3600.0 for v, (s,t) in timed.items()}
        first = sum(cost.values())
        replication = cost['R-OBS'] + max(cost[v] for v in CONTROLS)
        if first <= 16.0 and replication <= 12.0 and first+replication+other <= available:
            return n
    return None


@dataclass(frozen=True)
class Point:
    variant: str
    seed: int
    step: int
    ap: Mapping[int, float]
    current: Mapping[int, float]
    parameters: int
    complete: bool = True

    @property
    def long(self) -> float:
        return sum(self.ap[h] for h in (3,4,5))/3.0

    @property
    def current_long(self) -> float:
        return sum(self.current[h] for h in (3,4,5))/3.0

    def validate(self) -> None:
        if set(self.ap) != set(range(1,6)) or set(self.current) != set(range(1,6)):
            raise ValueError('CAL must contain all five horizons')
        if type(self.seed) is not int or type(self.step) is not int or self.step < 0:
            raise ValueError('invalid seed/update')
        if type(self.parameters) is not int or self.parameters < 0:
            raise ValueError('invalid parameter count')
        if self.complete is not True:
            raise ValueError('incomplete endpoint cannot be ranked')
        for k, val in list(self.ap.items()) + list(self.current.items()):
            finite(val, f'horizon {k}', high=1.0)


@dataclass(frozen=True)
class Selected:
    point: Point
    eligible: bool


def eligible(point: Point, m0: Point) -> bool:
    return (point.current_long >= m0.current_long-0.002
            and point.ap[2] >= m0.ap[2]-0.002)


def _metric_compare(a: Point, b: Point) -> int:
    for left, right in ((a.long, b.long), (a.current_long, b.current_long), (a.ap[2], b.ap[2])):
        if abs(left-right) > TOL:
            return -1 if left > right else 1
    return 0


def select_endpoint(points: Sequence[Point], m0: Point, *, n: int,
                    variant: str, seed: int) -> Selected:
    if n not in (400,800,1200) or variant not in VARIANTS:
        raise ValueError('invalid locked N or variant')
    m0.validate()
    if m0.variant != 'M0' or m0.step != 0:
        raise ValueError('guard requires the fresh M0 baseline')
    expected = (n//4, n//2, 3*n//4, n)
    if len(points) != 4 or sorted(p.step for p in points) != list(expected):
        raise ValueError('all four unique predeclared checkpoints are required')
    for p in points:
        p.validate()
        if (p.variant,p.seed) != (variant,seed):
            raise ValueError('mixed method or seed')
    admissible = [p for p in points if eligible(p,m0)]
    # Stable scan resolves tolerance-neighbour cases without relying on a
    # potentially non-transitive tolerance-based global sorting comparator.
    ordered = sorted(admissible or points, key=lambda p:p.step)
    best = ordered[0]
    for p in ordered[1:]:
        if _metric_compare(p,best) < 0:
            best = p
    return Selected(best, bool(admissible))


def select_control(selected: Mapping[str, Selected]) -> Selected:
    if set(selected) != set(CONTROLS):
        raise ValueError('all three fixed control families must be available')
    ordered = [selected[v] for v in sorted(CONTROLS)]
    for value in ordered:
        value.point.validate()
    def better(a: Selected, b: Selected) -> bool:
        if a.eligible != b.eligible:
            return a.eligible
        comparison = _metric_compare(a.point,b.point)
        if comparison:
            return comparison < 0
        return (a.point.parameters,a.point.variant) < (b.point.parameters,b.point.variant)
    best=ordered[0]
    for row in ordered[1:]:
        if better(row,best):
            best=row
    return best


def identity_weight(g_task: float, g_id: float) -> float:
    g_task=finite(g_task,'task gradient')
    g_id=finite(g_id,'identity gradient')
    if g_task <= 0 or g_id <= 0:
        raise ValueError('calibration requires positive finite gradient medians')
    values=(0.01,0.03,0.1,0.3)
    return min(values,key=lambda value:(abs(math.log10(value*g_id/g_task)+1.0),value))


def evaluation_jobs(n: int, control: str) -> list[dict[str, object]]:
    if n not in (400,800,1200) or control not in CONTROLS:
        raise ValueError('invalid locked N or control')
    jobs=[]
    def add(method: str, seed: int|None, step: int|str, role: str):
        jobs.append(dict(method=method,seed=seed,step=step,role=role))
    for method,seed,step in (('M0',None,0),('M4-200',146,200)):
        for role in ('CAL','SEL','PB'): add(method,seed,step,role)
    for seed,methods in ((246,VARIANTS),(247,('R-OBS',control))):
        for method in methods:
            for step in (n//4,n//2,3*n//4,n): add(method,seed,step,'CAL')
            for role in ('SEL','PB'): add(method,seed,'CAL_SELECTED',role)
    for role in ('SEL','PB'): add('FH-COMMIT0',None,0,role)
    if len(jobs) != 44:
        raise AssertionError('core evaluation matrix must contain 44 jobs')
    return jobs


def self_test() -> dict[str, object]:
    """Synthetic policy checks only; no research performance is measured."""
    checks=[]
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def rejects(name, callback):
        try: callback()
        except ValueError: checks.append(name); return
        raise AssertionError(name)
    def point(variant, step, long=.1, cur=.4, t2=.2, seed=246, params=10):
        return Point(variant,seed,step,{1:.4,2:t2,3:long,4:long,5:long},
                     {1:.4,2:cur,3:cur,4:cur,5:cur},params)
    base=point('M0',0)
    check('budget_does_not_reset', abs(new_budget(1760.667754119)-50)<1e-12)
    check('budget_protects_reserve', abs(new_budget(1810)-5)<1e-12)
    check('resume_does_not_renew_run_cap',new_budget(1760.667754119,own_spent=10)==40)
    check('parallel_reservations_deducted',new_budget(1810,active_reservations=3)==2)
    rejects('nonfinite_budget_rejected',lambda:new_budget(float('nan')))
    costs={v:(20.,17.) for v in VARIANTS}
    check('common_training_size_cost_only',select_n(costs,available=50,nontraining_forecast=16,evaluation_forecast=10)==400)
    check('infeasible_campaign_no_fake_run',select_n(costs,available=1,nontraining_forecast=16,evaluation_forecast=10) is None)
    rows=[point('R-OBS',s,.11 if s==200 else .1) for s in (100,200,300,400)]
    check('selects_full_cal_only',select_endpoint(rows,base,n=400,variant='R-OBS',seed=246).point.step==200)
    rejects('missing_checkpoint_not_winner',lambda:select_endpoint(rows[:-1],base,n=400,variant='R-OBS',seed=246))
    bad=[point('R-OBS',s,.2,cur=.39) for s in (100,200,300,400)]
    check('all_guard_fail_descriptive',not select_endpoint(bad,base,n=400,variant='R-OBS',seed=246).eligible)
    ties=[point('R-OBS',s) for s in (400,300,200,100)]
    check('earlier_checkpoint_tie',select_endpoint(ties,base,n=400,variant='R-OBS',seed=246).point.step==100)
    controls={v:Selected(point(v,200,params=i+1),True) for i,v in enumerate(CONTROLS)}
    check('control_tie_lower_parameters',select_control(controls).point.variant=='R-TARGET')
    check('lambda_from_train_gradient_ratio',identity_weight(1.,1.)==0.1)
    check('core_evaluation_jobs_44',all(len(evaluation_jobs(800,c))==44 for c in CONTROLS))
    check('two_seeds_not_inference_repeats',sum(j['seed']==247 for j in evaluation_jobs(800,'R-ID'))==12)
    return dict(status='PASS',scope='synthetic CPU policy checks, not model training',
                count=len(checks),checks=checks)

if __name__=='__main__':
    import json
    print(json.dumps(self_test(),ensure_ascii=False,indent=2))
