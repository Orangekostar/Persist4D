"""Check this instruction package, NOT the user's repository or server results.

Requires Python 3.10+, PyYAML, and CPU PyTorch. No network, GPU or user data.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import copy
import yaml
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent

def main() -> int:
    torch.set_num_threads(2)
    cfg = yaml.safe_load((ROOT / 'qp_mn_targeted_v1.design.yaml').read_text())
    md = (ROOT / 'ReScene_QP_MN_Codex_FINAL.md').read_text()
    rows = []
    def check(name: str, passed: bool, detail: str = '') -> None:
        rows.append({'check': name, 'passed': bool(passed), 'detail': detail})

    check('local_repaired_base_required', cfg['base']['requested_local_ref'] == '465f37a'
          and not cfg['base']['fallback_to_remote_old_labels_allowed'])
    check('required_local_reports', len(cfg['base']['required_reports']) == 2
          and all(p in md for p in cfg['base']['required_reports']))
    check('only_four_paired_arms', cfg['scope']['mandatory_paired_arms'] == ['Q_A0','Q_P','M_C','M_N'])
    check('no_combinations_or_formal_tests', not cfg['selection']['combinations']
          and cfg['scope']['formal_test_populations'] == [])
    check('one_module_per_prediction', cfg['scope']['maximum_active_modules_per_prediction'] == 1)
    check('frozen_parent', cfg['parent']['frozen'] and cfg['parent']['eval'])
    check('real_single_scan_T1', cfg['data']['t1'] == 'independent_single_scan_forward_unique_physical_scans')
    check('noaug_primary', cfg['data']['main_cal_sel_condition'] == 'NOAUG_NATIVE'
          and '本轮主评价明确改为无训练增强' in md)
    check('common_fixed_training_inputs', cfg['data']['train_inputs'] == 'repaired_controlled_training_parent_cache')
    check('qa0_qp_only_skip_difference',
          {k:v for k,v in cfg['quality']['arms']['Q_A0'].items() if k!='parent_score_skip'} ==
          {k:v for k,v in cfg['quality']['arms']['Q_P'].items() if k!='parent_score_skip'} and
          not cfg['quality']['arms']['Q_A0']['parent_score_skip'] and cfg['quality']['arms']['Q_P']['parent_score_skip'])
    check('no_extra_score_regularizer_or_clip', cfg['quality']['score_residual_regularization'] == 0
          and not cfg['quality']['clip_output'] and not cfg['quality']['rerun_topk_filter_nms'])
    check('unchanged_M_assignment_sampling', cfg['mask']['same_assignment_all_arms']
          and not cfg['mask']['new_multi_to_one'] and not cfg['mask']['new_resampling'])
    check('M_regularizer_not_scaled', cfg['mask']['regularization']['coefficient'] == .01
          and not cfg['mask']['regularization']['scale_with_positive_ratio'])
    check('same_bound_M_control', cfg['mask']['original_logit_residual_bound'] == 2.)
    check('steps_and_schedule', cfg['training']['updates'] == cfg['training']['scheduler_horizon'] == 1500
          and cfg['training']['checkpoints'] == [0,500,1000,1500])
    check('batch_16_slots', len(cfg['training']['records_per_update']) * cfg['training']['candidates_per_record'] == 16
          and cfg['training']['records_per_update'] == [2,2,2,1])
    check('CAL_train_points_only', cfg['selection']['selectable_steps'] == [500,1000,1500]
          and cfg['selection']['lock_all_before_SEL'])
    check('SEL_endpoint_no_selection', cfg['selection']['paired_SEL_fixed_endpoint'] == 1500
          and not cfg['selection']['endpoint_selectable_after_SEL'])
    check('fixed_replication_steps', cfg['training']['replication_stop'] == 'max_required_locked_step_not_new_epoch_search')
    check('budget_sum_16', math.isclose(sum(cfg['budget']['categories_gpu_hours'].values()),16.))
    check('prior_costs_not_reset', cfg['budget']['include_local_repair_costs']
          and cfg['budget']['prior_reference_is_not_current_balance'])
    check('bounded_cpu_gpu_checks', cfg['budget']['targeted_tests_wall_minutes'] == 20
          and cfg['budget']['gpu_smoke_target_hours'] == .25)
    check('small_metrics_can_publish_git', cfg['publication']['small_head_and_metric_package_direct_git']
          and cfg['publication']['all_required_assets_in_git_status'] == 'GIT_COMPLETE')
    check('no_GT_or_parent_redistribution', not cfg['publication']['raw_parent_weights_or_GT_publication'])
    check('source_index_complete', all(f'[E{i:02d}]' in md for i in range(1,11)))
    check('full_sections_present', all(f'## {i}.' in md for i in range(15)))
    check('code_not_claimed_implemented', '下面是待实现接口' in md)
    check('M_probe_deduplicated', 'M_FIT_DIAGNOSTIC' in md and '分母相同' in md)

    # Reference formulas, not imported repository implementations.
    torch.manual_seed(45)
    h = nn.Sequential(nn.Linear(8,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
    nn.init.zeros_(h[-1].weight); nn.init.zeros_(h[-1].bias)
    h_a = copy.deepcopy(h); h_p = copy.deepcopy(h)
    x = torch.randn(16,8)
    s0 = torch.linspace(.05,.95,16)
    y = torch.linspace(.8,.1,16)
    check('identical_QA0_QP_initial_tensors', all(torch.equal(a,b) for a,b in zip(h_a.state_dict().values(),h_p.state_dict().values())))
    ra = h_a(x).flatten(); rp = h_p(x).flatten()
    check('QP_zero_equals_B0', torch.equal(s0+rp,s0))
    check('QA0_zero_not_B0', torch.count_nonzero(ra).item()==0 and not torch.equal(ra,s0))
    opt = torch.optim.AdamW(h_p.parameters(),lr=.001,weight_decay=.0001)
    opt.zero_grad(); loss = ((s0+h_p(x).flatten()-y)**2).mean(); loss.backward()
    check('QP_output_first_step_has_gradient', h_p[-1].weight.grad.norm().item()>0)
    check('QP_trunk_first_step_zero_expected', h_p[0].weight.grad.norm().item()==0)
    opt.step(); opt.zero_grad(); loss = ((s0+h_p(x).flatten()-y)**2).mean(); loss.backward()
    check('QP_trunk_second_step_has_gradient', h_p[0].weight.grad.norm().item()>0)

    losses = torch.tensor([2.,3.]+[0.]*14,requires_grad=True)
    p = torch.tensor([True,True]+[False]*14)
    regularization = torch.tensor(.04,requires_grad=True)
    lc = losses.sum()/16 + .01*regularization
    ln = losses[p].sum()/p.sum() + .01*regularization
    check('MC_normalization_value', abs(lc.item()-.3129)<1e-6)
    check('MN_normalization_value', abs(ln.item()-2.5004)<1e-6)
    gc = torch.autograd.grad(lc,regularization,retain_graph=True)[0]
    gn = torch.autograd.grad(ln,regularization,retain_graph=True)[0]
    check('regularizer_gradient_unchanged', torch.equal(gc,gn) and abs(gn.item()-.01)<1e-8)
    gshape = torch.autograd.grad(ln,losses)[0]
    check('MN_only_positive_shape_gradients', torch.equal(gshape,torch.tensor([.5,.5]+[0.]*14)))
    delta = torch.ones(16,requires_grad=True)
    empty_loss = 0.*delta.sum() + .01*delta.square().mean()
    empty_loss.backward()
    check('zero_positive_finite_regularizer', abs(empty_loss.item()-.01)<1e-7
          and torch.isfinite(delta.grad).all().item() and (delta.grad>0).all().item())
    allpos = torch.arange(1.,17.)
    check('all_positive_formulas_identical', torch.equal(allpos.sum()/16, allpos.mean()))
    z = torch.tensor([-3.,3.]); r = 2*torch.tanh(torch.tensor([100.,-100.]))
    check('bounded_residual_cannot_flip_strong_errors', torch.equal(z>0, z+r>0))
    scores = torch.tensor([-.7,1.3,.2,2.1])
    check('finite_real_score_affine_order', torch.equal(torch.argsort(scores),torch.argsort(2*scores+4)))
    check('AP_unit_conversion', math.isclose(.001*100,.1))
    check('historical_supervised_fraction_derivation', math.isclose(2223/24000*100,9.2625))
    result = {'scope':'instruction package static checks and synthetic CPU formula examples only',
              'not_executed':['user_repository_regression','stmetrics_evaluation','R1_forward','GPU_training','GitHub_push'],
              'checks': rows,'passed':sum(r['passed'] for r in rows),'failed':sum(not r['passed'] for r in rows)}
    (ROOT/'REVIEW_CHECKS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('scope','passed','failed')},ensure_ascii=False))
    return 1 if result['failed'] else 0

if __name__=='__main__':
    raise SystemExit(main())
