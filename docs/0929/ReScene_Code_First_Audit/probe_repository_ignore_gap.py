"""Read-only CPU probe for the actual Persist4D repository and installed stmetrics.

This script was prepared, NOT executed in the user's server environment.
It creates a tiny synthetic two-scan example. Optionally counts affected label
records in an existing EXPORT_INDEX cache. It does not train, change thresholds,
modify the dataset, patch the repository, or claim a real-data AP improvement.

Usage in the existing Persist4D Python environment:
  python probe_repository_ignore_gap.py --repo /path/to/Persist4D
  python probe_repository_ignore_gap.py --repo /path/to/Persist4D \
      --cache-root /path/to/short_module_screen_v1
"""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys
import tempfile
import torch


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--cache-root', type=Path)
    args = p.parse_args()
    repo = args.repo.resolve()
    if not (repo/'scripts/short_module_data.py').is_file():
        raise SystemExit('Repository does not contain scripts/short_module_data.py')
    sys.path.insert(0, str(repo))
    from scripts.short_module_data import quality_labels
    from scripts.p6a_metrics import OfficialMetricAccumulator
    from stmetrics.instances.matcher import InstanceMatcher
    import stmetrics.instances.evaluator as evaluator_module
    torch.set_num_threads(1)
    torch.manual_seed(45)

    with tempfile.TemporaryDirectory(prefix='rescene-ignore-audit-') as tmp:
        spec = Path(tmp)/'spec.yaml'
        spec.write_text('name: rio\nclass_labels: [chair]\nvalid_class_ids: [3]\n'
                        'aux: changes\naux_labels: [static]\nvalid_aux_ids: [0]\n')
        stages = torch.tensor([0]*250 + [1]*250)
        gt_mask = torch.zeros(500, dtype=torch.bool)
        gt_mask[:100] = True
        gt_mask[250:350] = True
        bad_mask = torch.ones(500, dtype=torch.bool)
        target = {'masks': gt_mask[None], 'labels':torch.tensor([3]),
                  'ids':torch.tensor([11]), 'changes':torch.tensor([0]),
                  'temporal_stages':stages}
        pred = {'pred_masks':torch.stack([bad_mask, gt_mask], dim=1),
                'pred_scores':torch.tensor([.1,.9]), 'pred_classes':torch.tensor([3,3])}
        labels = quality_labels(pred,target,dataset_spec=spec,min_region_size=100)
        accumulator=OfficialMetricAccumulator(mode='strict_online',dataset_spec=spec,min_region_size=100)
        matcher=accumulator._metric.matcher
        evaluator=accumulator._metric.heads[0]
        cfg=matcher.config
        records=[]
        for tau in (.5,.75):
            for bad_score in (.1,.95):
                current={**pred,'pred_scores':torch.tensor([bad_score,.9])}
                gt2pred,pred2gt=matcher.match_batch([current],[target])[0]
                params=(evaluator.label,tau,int(cfg.min_region_sizes[0]),
                        float(cfg.distance_threshes[0]),float(cfg.distance_confs[0]),'chair')
                truth,scores,hard_fn,_=evaluator._evaluate_class(gt2pred,pred2gt,params)
                records.append({'threshold':tau,'bad_score':bad_score,
                    'scored_y_true':truth.tolist(),'scored_y_score':scores.tolist(),
                    'hard_fn':hard_fn})
        ignore_value=labels['ignore_proportion'][0]
        result={'scope':'Actual repository functions and installed evaluator, synthetic input only',
            'bad_candidate': {'mask_points_per_stage':250,'GT_points_per_stage':100,
                'analytical_temporal_iou':.4,'ignore_proportion':float(ignore_value),
                'training_status':labels['status'][0], 'training_valid':bool(labels['valid'][0])},
            'official_evaluation':records,
            'source_blobs':{'label_adapter':git_blob_sha(repo/'scripts/short_module_data.py'),
                'matcher':git_blob_sha(Path(inspect.getfile(InstanceMatcher))),
                'evaluator':git_blob_sha(Path(inspect.getfile(evaluator_module)))} }
        result['gap_reproduced']=(not result['bad_candidate']['training_valid']
            and any(r['threshold']==.75 and len(r['scored_y_true'])==2 and 0 in r['scored_y_true']
                    for r in records))

    if args.cache_root:
        root=args.cache_root.resolve()
        index=json.loads((root/'EXPORT_INDEX.json').read_text())
        cache=Path(index['cache'])
        summaries=defaultdict(Counter)
        for entry in index['entries']:
            # Only user-selected, local, trusted experiment cache files are loaded.
            target_pack=torch.load(cache/entry['targets']['file'],map_location='cpu',weights_only=False)
            quality=target_pack['quality']
            ts=[float(t) for t in torch.tensor(quality['thresholds'],dtype=torch.float32)]
            key=(entry['record']['role'],entry['record']['horizon'])
            count=summaries[key]
            count['candidates']+=len(quality['status'])
            for i,status in enumerate(quality['status']):
                count['status_'+status]+=1
                ignore=quality['ignore_proportion'][i]
                if ignore is None:
                    continue
                ig=float(ignore)
                if status=='IGNORE' and min(ts)<ig<=max(ts):
                    count['excluded_but_potentially_scored_at_higher_thresholds']+=1
                    for tau in ts:
                        if ig<=tau:
                            count[f'excluded_but_scored_if_unmatched_tau_{tau:.2f}']+=1
        result['existing_cache_coverage']=[{'role':role,'H':h,**dict(c)} for (role,h),c in sorted(summaries.items())]
        result['coverage_note']='Counts label/evaluation eligibility, not actual reranked FP causality or AP gain.'
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
