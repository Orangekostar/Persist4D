"""CPU counterexamples for the inspected code, not a full repository/GPU run.

Sources: Persist4D@4bf00902c9428795f7f47547bf462540aa304cc6
  scripts/short_module_data.py::quality_labels
  models/short_module_heads.py::MaskHead, QualityHead
  scripts/short_module_evaluation.py::evaluate_point
  scripts/short_module_screen.py::run_pipeline
Official evaluator blob: 6c558fad20ecc033c1763864732eb98f2902108e.
Only the relevant predicates/equations are reconstructed below; this file does
not claim to import the deployed repository or reproduce its real-data AP.
"""
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
THRESHOLDS = (.50,.55,.60,.65,.70,.75,.80,.85,.90)
results = {}

# 1. Per-stage masks all have >=100 points, so no region-size filtering changes
#    this example: GT=100 points/stage, bad prediction=250 points/stage and
#    includes the whole GT. Ignore proportion=150/250; tIoU=100/250.
ignore_proportion = 150 / 250
bad_tiou = 100 / 250
training_valid = not (ignore_proportion > min(THRESHOLDS))
# Official evaluator: unsuccessful candidates remain scored iff p_ignore<=tau.
official_scored = {str(t): ignore_proportion <= t for t in THRESHOLDS}
assert not training_valid
assert not official_scored['0.5'] and official_scored['0.75']
assert bad_tiou < .5
results['ignore_threshold_gap'] = {
    'training_label_valid': training_valid,
    'ignore_proportion': ignore_proportion,
    'temporal_iou': bad_tiou,
    'official_scored_as_false_positive_by_threshold': official_scored,
    'finding': 'Whole-candidate exclusion at 0.50 loses labels for errors scored at higher thresholds.'}

# One correct candidate plus the bad candidate. This reproduces the relevant
# score-order and ignore predicates of Evaluator._match, not full AP math.
def match_core(scores, tau):
    candidates = [('good', scores[0], 1., 0.), ('bad', scores[1], bad_tiou, ignore_proportion)]
    matched = False
    scored = []
    for name, score, overlap, p_ignore in sorted(candidates, key=lambda r:r[1], reverse=True):
        positive = not matched and overlap > tau
        if positive:
            matched = True
        if positive or p_ignore <= tau:
            scored.append({'candidate':name, 'score':score, 'tp':positive})
    return scored
base = match_core((.9,.1),.75)
changed = match_core((.9,.95),.75)
assert [r['tp'] for r in base] == [True,False]
assert [r['tp'] for r in changed] == [False,True]
results['ignore_gap_ranking_example'] = {'at_075_original':base, 'at_075_reranked':changed,
    'note':'Illustrates an AP-relevant ordering change; no real-data AP was computed.'}

# 2. Model eval does not change an independent dataset.mode string.
network=nn.Linear(2,2).eval(); dataset_mode='train'
assert network.training is False and 'train' in dataset_mode
results['data_mode_independence'] = {'network_training':network.training,
    'dataset_mode':dataset_mode, 'augmentation_branch_taken': 'train' in dataset_mode}

# 3. Head equations from the inspected implementation: zero last layer is
#    differentiable and only delays the hidden-layer gradient by one update.
class MaskHeadCore(nn.Module):
    def __init__(self, d=4, dq=3, classes=2):
        super().__init__(); self.d=d
        self.trunk=nn.Sequential(nn.Linear(dq+2*d+classes,128),nn.GELU())
        self.output=nn.Linear(128,d+1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
    def forward(self, features, query, h, onehot, stages):
        inputs=torch.cat((query[:,None,:].expand(-1,h.shape[1],-1),h,
                          onehot[:,None,:].expand(-1,h.shape[1],-1)),-1)
        coeff=self.output(self.trunk(inputs))
        a=coeff[:,stages,:-1].permute(1,0,2)
        b=coeff[:,stages,-1].transpose(0,1)
        return 2*((features[:,None,:]*a).sum(-1)/math.sqrt(self.d)+b).tanh()
torch.manual_seed(45)
head=MaskHeadCore(); opt=torch.optim.AdamW(head.parameters(),lr=1e-3)
x=torch.randn(6,4); q=torch.randn(2,3); h=torch.randn(2,2,8)
onehot=torch.eye(2); stages=torch.tensor([0,0,0,1,1,1]); z=torch.randn(6,2)
y=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[1.,0.],[0.,1.],[0.,1.]])
assert torch.equal(head(x,q,h,onehot,stages),torch.zeros(6,2))
grad=[]
for _ in range(2):
    opt.zero_grad(); delta=head(x,q,h,onehot,stages)
    loss=F.binary_cross_entropy_with_logits(z+delta,y)+.01*delta.square().mean()
    loss.backward()
    grad.append({'output_gradient_norm':head.output.weight.grad.norm().item(),
                 'trunk_gradient_norm':head.trunk[0].weight.grad.norm().item()})
    opt.step()
assert grad[0]['output_gradient_norm']>0 and grad[1]['trunk_gradient_norm']>0
results['mask_gradient_and_zero_init']={'passed':True, 'steps':grad,
    'scope':'equations and gradient path only; no real R1 data or trained head loaded'}

# 4. Q3 monotone transformation is not a detached/hard-cummin path.
raw=torch.randn(3,9,requires_grad=True)
inc=torch.cat((raw[:,:1],-F.softplus(raw[:,1:])),dim=-1)
p=inc.cumsum(-1).sigmoid()
F.binary_cross_entropy(p,torch.zeros_like(p)).backward()
assert bool((p[:,1:]<=p[:,:-1]).all()) and bool(torch.isfinite(raw.grad).all())
assert int((raw.grad!=0).sum())==raw.numel()
results['q3_monotonic_gradients']={'passed':True,'nonzero_gradient_elements':int((raw.grad!=0).sum())}

# 5. Existing result-reuse predicate does not compare the CURRENT helper code.
previous={'status':'COMPLETE','cache_identity_sha256':'old-cache','head_sha256':'same-head'}
index={'cache_identity_sha256':'old-cache'}
reuse=(previous['status']=='COMPLETE' and previous['cache_identity_sha256']==index['cache_identity_sha256']
       and previous['head_sha256']=='same-head')
helper_before='return head(value)'; helper_after='return head(value) + 0.001'
assert helper_before!=helper_after and reuse
results['helper_cache_invalidation_gap']={'old_result_reused_by_current_predicate':reuse,
    'helper_changed':True,'scope':'demonstrated reachable condition; not evidence it occurred in the recorded campaign'}

# 6. Bounded residual is an intentional method restriction, not a gradient bug.
large=torch.tensor([-3.,3.]); attempted=2*torch.tanh(torch.tensor([100.,-100.]))
assert torch.equal((large>0),(large+attempted>0))
results['bounded_residual_limitation']={'parent_logits':large.tolist(),
    'largest_attempted_correction':attempted.tolist(),'corrected_logits':(large+attempted).tolist(),
    'classification':'design constraint, not code defect'}

report={'torch_version':torch.__version__,'device':'cpu','all_assertions_passed':True,
        'scope':'minimal counterexamples and equations, not full repository tests or GPU rescoring',
        'results':results}
Path(__file__).with_name('minimal_test_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
