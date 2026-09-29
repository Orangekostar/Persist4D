"""At most one prescribed fixed-candidate probe per triggered diagnostic."""
from pathlib import Path

import torch
from torch.nn import functional as F

from models.qp_mn_heads import build_head
from scripts.qp_mn_adapter import batch_loss
from scripts.qp_mn_binding import gpu_scope
from scripts.qp_mn_diagnostics import ordinary_iou
from scripts.rescene_task_postprocess import materialize_segment_logits
from scripts.short_module_evaluation import materialization_system, write_csv
from scripts.short_module_identity import file_digest, identity_digest, source_identity
from scripts.short_module_native import unpack_prediction
from scripts.short_module_screen import append_event, read_json, write_json
from scripts.short_module_training import load_training_records
from scripts.system_comparison_inference import unpack_bool_matrix


def make_fit_draws(rows, records):
    selected = sorted(rows, key=lambda r:(r['input_id'],r['source_query_id'],r['source_class_id'],r['retained_index']))[:16]
    if len(selected) < 8:
        raise ValueError('M fit requires at least eight predeclared eligible candidates')
    draws = []
    for n, row in enumerate(selected):
        r, i = records[row['input_id']], row['retained_index']
        if not r['quality']['geometry_valid'][i] or r['assignment'][i] < 0 or not (r['weights'] > 0).any():
            raise ValueError('fixed probe candidate lacks actual shape supervision; do not resample')
        draws.append({'input_id':row['input_id'], 'reference_id':row['reference_id'], 'horizon':row['H'],
                      'candidates':[i], 'segment_seed':145+n})
    return draws


def m_geometry(context, head, draws, *, device, step):
    c = context
    index = c.train_index
    entries = {e['input_id']:e for e in index['entries']}
    system = materialization_system(index)
    rows = []
    with torch.no_grad():
        for draw in draws:
            i = draw['candidates'][0]
            entry = entries[draw['input_id']]
            saved = torch.load(Path(index['cache'])/entry['prediction']['file'], map_location='cpu', weights_only=False)
            target = torch.load(Path(index['cache'])/entry['targets']['file'], map_location='cpu', weights_only=False)
            parent = unpack_prediction(saved['parent'])
            soft, d = parent.soft_evidence, saved['descriptor']
            gt_masks = unpack_bool_matrix(target['target']['masks'])
            gt = gt_masks[target['target']['ids'] == target['assignment'][i]].any(0)
            delta = head(soft.segment_features.to(device),soft.query_features[i:i+1].to(device),d['h'][i:i+1].to(device),
                         F.one_hot(soft.source_class_ids[i:i+1].to(device),d['classes']).float(),d['segment_stages'].to(device)).cpu()
            logits = soft.segment_logits.clone()
            logits[:,i] += delta[:,0]
            materialized = materialize_segment_logits(system=system,evidence=soft,segment_logits=logits)[:,i]
            raw = logits[soft.low_point2segment[soft.voxel_inverse],i] > 0
            rows.append({'input_id':draw['input_id'],'retained_index':i,'step':step,'status':'DIAGNOSTIC_ONLY',
                         'raw_expanded_iou':ordinary_iou(raw,gt),'materialized_iou':ordinary_iou(materialized,gt),
                         'original_materialized_iou':ordinary_iou(parent.pred_masks[:,i],gt),
                         'delta_abs_mean':float(delta.abs().mean())})
    return rows


def run_probes(context, *, device='cuda:0'):
    c = context
    diagnostics = read_json(c.artifacts/'diagnostics/DIAGNOSTICS.json')
    result = {'Q_MEMORIZATION':{'status':'NOT_APPLICABLE','trigger':diagnostics['q_memorization_trigger']},
              'M_FIT_DIAGNOSTIC':{'status':'NOT_APPLICABLE','trigger':diagnostics['m_fit_trigger'],
                                  'eligible_candidates':diagnostics['m_fit_eligible_candidates']}}
    if not diagnostics['m_fit_trigger'] and not diagnostics['q_memorization_trigger']:
        write_json(c.artifacts/'diagnostics/PROBES.json',result)
        return result
    records = load_training_records(c.root,c.train_index)
    tasks = []
    if diagnostics['m_fit_trigger']:
        tasks.append(('M_FIT_DIAGNOSTIC','M_N',make_fit_draws(diagnostics['m_fit_candidates'],records),500,(0,250,500)))
    if diagnostics['q_memorization_trigger']:
        panel = read_json(c.artifacts/'diagnostics/TRAIN_PANEL.json')['records']
        candidates = []
        for r in panel:
            cached = records[r['input_id']]
            entry = next(e for e in c.train_index['entries'] if e['input_id']==r['input_id'])
            saved = torch.load(Path(c.train_index['cache'])/entry['prediction']['file'],map_location='cpu',weights_only=False)
            for i in torch.where(cached['quality']['valid'])[0].tolist():
                candidates.append((tuple(saved['candidate_keys'][i]),r,i))
        candidates = sorted(candidates,key=lambda r:r[0])[:16]
        if len(candidates)!=16:
            raise ValueError('Q memorization requires fixed16 supervisable TRAIN candidates')
        draws = [{'input_id':r['input_id'],'reference_id':r['reference_id'],'horizon':r['horizon'],
                  'candidates':[i],'segment_seed':145+n} for n,(_,r,i) in enumerate(candidates)]
        tasks.append(('Q_MEMORIZATION','Q_P',draws,200,(0,100,200)))
    with gpu_scope(c,device=device,stage='conditional-fixed-probes'):
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        for name,mode,draws,updates,checkpoints in tasks:
            head=build_head(mode,**c.dimensions,seed=145).to(device)
            optimizer=torch.optim.AdamW(head.parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=.0001)
            metadata={'status':'DIAGNOSTIC_ONLY','name':name,'seed':145,'mode':mode,'updates':updates,
                      'fixed_candidates':draws,'sample_sha256':identity_digest(draws),
                      'parent_sha256':c.config['parent_sha256'],'label_identity_sha256':c.train_index['label_identity_sha256'],
                      'sources':source_identity([Path(__file__)]),'main_selection_eligible':False}
            directory=c.root/'diagnostic_probes'/name
            public=c.artifacts/'diagnostics/probes'/name
            write_json(public/'CONFIG.json',metadata)
            geometry, losses=[] ,[]
            for step in range(updates+1):
                if step in checkpoints:
                    directory.mkdir(parents=True,exist_ok=True)
                    path=directory/f'update={step:04d}.pt'
                    torch.save({'head':{k:v.detach().cpu() for k,v in head.state_dict().items()},
                                'optimizer':optimizer.state_dict(),'metadata':{**metadata,'step':step},
                                'torch_rng':torch.get_rng_state()},path)
                    if mode=='M_N':
                        geometry.extend(m_geometry(c,head,draws,device=device,step=step))
                if step==updates:
                    break
                optimizer.zero_grad(set_to_none=True)
                loss,detail,_=batch_loss(mode,head,records,draws,classes=c.dimensions['classes'],device=device)
                if detail['B']!=detail['Npos'] or not torch.isfinite(loss):
                    raise ValueError('fixed probe has invalid or unsupervised positions')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                losses.append(loss.item())
                if (step+1)%100==0:
                    append_event(public/'metrics.jsonl',{'step':step+1,'loss':loss.item(),**detail,'preclip_norm':norm.item()})
            terminal,_,_=batch_loss(mode,head,records,draws,classes=c.dimensions['classes'],device=device)
            write_csv(public/'GEOMETRY.csv',geometry)
            result[name]={**metadata,'status':'DIAGNOSTIC_ONLY','first_training_loss':losses[0],
                          'terminal_loss':float(terminal.detach()),'geometry_rows':len(geometry),
                          'checkpoints':[{'path':str(p),'sha256':file_digest(p),'bytes':p.stat().st_size}
                                         for p in sorted(directory.glob('update=*.pt'))]}
    write_json(c.artifacts/'diagnostics/PROBES.json',result)
    return result
