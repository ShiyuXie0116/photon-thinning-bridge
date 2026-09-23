"""Evaluation across the dose sweep, including off-grid doses."""

import os, sys, json, argparse
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified_eval import UNetWithTime, compute_psnr, compute_ssim
from baselines.train_baselines import REDCNN, split_slice_ids, OUT_ROOT
from baselines.eval_all import try_lpips, lpips_val
from baselines.train_bridge import UNetRes, REDCNNNoReLU, build as build_bridge

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KNOTS = {
    '2detect': {
        'uniform': f"{DATA_ROOT}/2DeteCT_dose_bridge/precomputed",
        'EI': f"{DATA_ROOT}/2DeteCT_dose_bridge/precomputed_equal_improvement",
        'B': f"{DATA_ROOT}/2DeteCT_dose_bridge/alt_refs",   # fresh t=1 realization (x1B), ref x0_pre
        'offgrid': f"{DATA_ROOT}/2DeteCT_dose_bridge_offgrid/precomputed",  # test-only, 7 off-grid doses
    },
    'ldct': {
        'EI': f"{DATA_ROOT}/LDCT_dose_bridge/precomputed_equal_improvement",
    },
}
BASE = {'2detect': f"{DATA_ROOT}/2DeteCT_dose_bridge",
        'ldct': f"{DATA_ROOT}/LDCT_dose_bridge"}


def registry(ds):
    """name -> (ckpt, arch). arch in {unet_t, unet_ep, unet_res, redcnn, redcnn_res0}."""
    b = BASE[ds]
    ck = 'x0pred_best.pth' if ds == '2detect' else 'ldct_dose_bridge_best.pth'
    ckb = 'dose_bridge_best.pth' if ds == '2detect' else 'ldct_dose_bridge_best.pth'
    R = {
        'x0_uniform': (f'{b}/output_x0pred/{ck}', 'unet_t'),
        'x0_dose_unif': (f'{b}/output_geometric_x0pred/{ckb}', 'unet_t'),
        'x0_EI': (f'{b}/output_equal_improvement_x0pred/{ckb}', 'unet_t'),
        'endpoint_unet': (f'{OUT_ROOT}/{ds}_unet_endpoint/best.pth', 'unet_ep'),
        'endpoint_redcnn': (f'{OUT_ROOT}/{ds}_redcnn_endpoint/best.pth', 'redcnn'),
        'bridge_redcnn_uniform': (f'{OUT_ROOT}/{ds}_redcnn_bridge_uniform/best.pth', 'redcnn'),
        'bridge_redcnn_EI': (f'{OUT_ROOT}/{ds}_redcnn_bridge_equal_improvement/best.pth', 'redcnn'),
    }
    for arch in ('unet', 'unet_res', 'unet_res_nn', 'redcnn', 'redcnn_res0',
                 'drunet', 'nafnet', 'drunet_blind', 'nafnet_blind', 'dncnn', 'redcnn_wide', 'redcnn_t', 'redcnn_wide_t'):
        a = {'unet': 'unet_t', 'unet_res': 'unet_res', 'unet_res_nn': 'unet_res_nn',
             'redcnn': 'redcnn', 'redcnn_res0': 'redcnn_res0'}.get(arch, 'r6:' + arch)
        for sched in ('uniform', 'equal_improvement', 'geometric'):
            R[f'{arch}_bridge_{sched}'] = (f'{OUT_ROOT}/{ds}_{arch}_bridge_{sched}/best.pth', a)
            R[f'{arch}_endpoint_{sched}'] = (f'{OUT_ROOT}/{ds}_{arch}_endpoint_{sched}/best.pth', a)
            R[f'{arch}_interp_{sched}'] = (f'{OUT_ROOT}/{ds}_{arch}_interp_{sched}/best.pth', a)
            for k in range(4):
                R[f'{arch}_spec_{sched}_k{k}'] = (f'{OUT_ROOT}/{ds}_{arch}_endpoint_{sched}_k{k}/best.pth', a)
        R[f'{arch}_rep5'] = (f'{OUT_ROOT}/{ds}_rep5_{arch}_bridge_uniform/best.pth', a)
        for sched in ('uniform', 'equal_improvement'):
            for ep in (200, 300):
                R[f'{arch}_bridge_{sched}_ep{ep}'] = (f'{OUT_ROOT}/{ds}_{arch}_bridge_{sched}_ep{ep}/best.pth', a)
            R[f'{arch}_endpoint_{sched}_ep1000'] = (f'{OUT_ROOT}/{ds}_{arch}_endpoint_{sched}_ep1000/best.pth', a)
        for frac in ('0.05', '0.1', '0.25'):
            for mode in ('bridge', 'endpoint', 'interp'):
                R[f'{arch}_{mode}_uniform_frac{frac}'] = (f'{OUT_ROOT}/{ds}_{arch}_{mode}_uniform_frac{frac}/best.pth', a)
    archmap = {'unet': 'unet_t', 'unet_res': 'unet_res', 'unet_res96': 'unet_res96', 'unet_res48': 'unet_res48',
               'unet_res_nn': 'unet_res_nn', 'redcnn': 'redcnn', 'redcnn_res0': 'redcnn_res0'}
    archs = sorted(['unet', 'unet_res', 'unet_res96', 'unet_res48', 'unet_res_nn', 'redcnn', 'redcnn_res0', 'drunet', 'nafnet',
                    'drunet_blind', 'nafnet_blind', 'dncnn', 'redcnn_wide', 'redcnn_t', 'redcnn_wide_t', 'edcnn',
                    'unet_res_ref', 'unet_res_ref_w', 'unet_res96_ref', 'hybrid', 'hybrid96', 'wide_film', 'wganvgg', 'mapnn', 'restormer'], key=len, reverse=True)
    from glob import glob as _glob
    for ck in _glob(f'{OUT_ROOT}/{ds}_*/best.pth'):
        name = os.path.basename(os.path.dirname(ck))[len(f'{ds}_'):]
        if name.endswith('_smoke') or not os.path.exists(os.path.join(os.path.dirname(ck), 'test_summary.json')):
            continue
        exp_tag = ''
        for tag in ('rep4_', 'rep5_', 'rep8_'):
            if name.startswith(tag):
                exp_tag, name = tag[:-1], name[len(tag):]
        if name.startswith(('effI0_', 'realI0_')):
            continue
        arch = next((a for a in archs if name.startswith(a + '_')), None)
        if arch is None:
            continue
        key = '' + (exp_tag + '_' if exp_tag else '') + name
        if key not in R and not any(v[0] == ck for v in R.values()):
            R[key] = (ck, archmap.get(arch, 'r6:' + arch))
    def _done(ck):
        d = os.path.dirname(ck)
        return os.path.exists(ck) and (not os.path.basename(d).startswith('') or os.path.exists(os.path.join(d, 'test_summary.json')))
    return {n: v for n, v in R.items() if _done(v[0])}


def load_model(ckpt, arch, device):
    sd = torch.load(ckpt, map_location=device)
    sd = sd['model'] if isinstance(sd, dict) and 'model' in sd else sd
    if arch == 'unet_t':
        m = UNetWithTime(base_ch=64, t_dim=128); timed = True
    elif arch == 'unet_ep':
        m = UNetWithTime(base_ch=64, t_dim=128); timed = 'fixed1'
        sd = {k[4:] if k.startswith('net.') else k: v for k, v in sd.items()}
    elif arch == 'unet_res':
        m = UNetRes(); timed = True
    elif arch == 'unet_res96':
        m = UNetRes(base_ch=96); timed = True
    elif arch == 'unet_res48':
        m = UNetRes(base_ch=48); timed = True
    elif arch == 'unet_res_nn':
        m = UNetRes(no_norm=True); timed = True
    elif arch == 'redcnn':
        m = REDCNN(); timed = False
    elif arch == 'redcnn_res0':
        m = REDCNNNoReLU(); timed = False
    elif arch.startswith('r6:'):
        m, timed = build_bridge(arch[3:])
    else:
        raise ValueError(arch)
    m.load_state_dict(sd)
    return m.to(device).eval(), timed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['2detect', 'ldct'])
    ap.add_argument('--knots', default='all', help='comma list of knot tags or all')
    ap.add_argument('--models', default='all', help='comma list of model names or all')
    ap.add_argument('--out', default=None)
    ap.add_argument('--no_lpips', action='store_true')
    ap.add_argument('--save_pred', action='store_true', help='save predictions (float16) under out/preds/')
    ap.add_argument('--save_knots', default='uniform_step4,EI_step4,B_step4',
                    help='knots whose predictions are saved with --save_pred (comma list or all)')
    args = ap.parse_args()
    device = torch.device('cuda')
    ds = args.dataset
    out_dir = args.out or os.path.join(OUT_ROOT, f'eval_sweep_{ds}')
    os.makedirs(out_dir, exist_ok=True)

    knots = {k: v for k, v in KNOTS[ds].items()
             if (args.knots == 'all' or k in args.knots.split(',')) and os.path.isdir(v)}
    reg = registry(ds)
    if args.models != 'all':
        reg = {n: v for n, v in reg.items() if n in args.models.split(',')}
    print(f"dataset={ds}  knots={list(knots)}  models={list(reg)}", flush=True)

    data, knot_t = {}, {}
    for tag, d in knots.items():
        if tag == 'B':
            key = 'B_step4'; data[key] = {}
            for f in sorted(glob(os.path.join(d, 'slice*.npz'))):
                z = np.load(f); sid = os.path.basename(f)[:-4]
                data[key][sid] = (z['x1B'].astype(np.float32), z['x0_pre'].astype(np.float32))
            knot_t[key] = 1.0
            print(f"  {key}: t=1.000  n={len(data[key])} (fresh realization B)", flush=True)
            continue
        files = sorted(glob(os.path.join(d, '*_step*.npz')))
        sids = sorted({os.path.basename(f).split('_step')[0] for f in files})
        test = sids if tag == 'offgrid' else split_slice_ids(sids, ds, 'test')
        n_steps = 1 + max(int(os.path.basename(f).split('_step')[1].split('.')[0]) for f in files)
        for k in range(n_steps):
            key = f'{tag}_step{k}'
            data[key] = {}
            for sid in test:
                f = os.path.join(d, f'{sid}_step{k}.npz')
                if not os.path.exists(f):
                    continue
                z = np.load(f)
                xt, x0 = z['x_t'].astype(np.float32), z['x0'].astype(np.float32)
                if np.isfinite(xt).all() and np.isfinite(x0).all():
                    data[key][sid] = (xt, x0)
                    knot_t[key] = float(z['t'])
            print(f"  {key}: t={knot_t.get(key, float('nan')):.3f}  n={len(data[key])}", flush=True)

    lp = None if args.no_lpips else try_lpips(device)
    per = {}

    def record(method, key, pred, gt):
        e = per.setdefault(method, {}).setdefault(key, {'psnr': [], 'ssim': [], 'lpips': []})
        e['psnr'].append(compute_psnr(pred, gt)); e['ssim'].append(compute_ssim(pred, gt))
        v = lpips_val(lp, pred, gt, device)
        if v is not None:
            e['lpips'].append(v)

    for key, dd in data.items():
        for sid, (xt, x0) in dd.items():
            record('input', key, xt, x0)

    for name, (ckpt, arch) in reg.items():
        try:
            model, timed = load_model(ckpt, arch, device)
        except Exception as e:
            print(f"SKIP {name}: {e}"); continue
        variants = [('@t', 'true'), ('@t1', 'one')] if timed is True else [('', None)]
        preds = {}
        for key, dd in data.items():
            t_true = knot_t[key]
            for sid, (xt, x0) in tqdm(dd.items(), desc=f'{name} {key}', leave=False):
                x = torch.from_numpy(xt)[None, None].to(device)
                with torch.no_grad():
                    for suffix, tmode in variants:
                        if timed is False:
                            out = model(x)
                        else:
                            tv = 1.0 if (timed == 'fixed1' or tmode == 'one') else t_true
                            out = model(x, torch.tensor([tv], dtype=torch.float32, device=device))
                        pred = np.clip(out.cpu().numpy().squeeze(), 0, 1)
                        record(name + suffix, key, pred, x0)
                        if args.save_pred and (args.save_knots == 'all' or key in args.save_knots.split(',')):
                            preds[f'{name}{suffix}|{key}|{sid}'] = pred.astype(np.float16)
        if args.save_pred and preds:
            os.makedirs(os.path.join(out_dir, 'preds'), exist_ok=True)
            np.savez_compressed(os.path.join(out_dir, 'preds', f'{name}.npz'), **preds)
        del model; torch.cuda.empty_cache()
        print(f"done {name}", flush=True)

    summary = {'dataset': ds, 'knot_t': knot_t, 'n': {k: len(v) for k, v in data.items()}, 'models': {}}
    for m, byk in per.items():
        summary['models'][m] = {}
        for key, e in byk.items():
            s = {'psnr': float(np.mean(e['psnr'])), 'psnr_std': float(np.std(e['psnr'])),
                 'ssim': float(np.mean(e['ssim'])), 'n': len(e['psnr'])}
            if e['lpips']:
                s['lpips'] = float(np.mean(e['lpips']))
            summary['models'][m][key] = s
    pairs = [('unet_res_bridge_uniform@t', 'unet_res_endpoint_uniform@t1'),
             ('unet_res_bridge_uniform@t', 'endpoint_redcnn'),
             ('unet_res_bridge_equal_improvement@t', 'endpoint_redcnn'),
             ('unet_bridge_uniform@t', 'unet_endpoint_uniform@t1'),
             ('redcnn_bridge_uniform', 'redcnn_endpoint_uniform'),
             ('bridge_redcnn_uniform', 'endpoint_redcnn'), ('bridge_redcnn_EI', 'endpoint_redcnn'),
             ('x0_uniform@t', 'endpoint_unet'), ('x0_EI@t', 'endpoint_unet'),
             ('x0_uniform@t', 'endpoint_redcnn'),
             ('redcnn_t_bridge_uniform@t', 'redcnn_endpoint_uniform'),
             ('redcnn_t_bridge_uniform@t', 'endpoint_redcnn'),
             ('redcnn_t_bridge_equal_improvement@t', 'redcnn_endpoint_uniform'),
             ('drunet_bridge_uniform@t', 'drunet_endpoint_uniform@t1'),
             ('drunet_bridge_uniform@t', 'redcnn_endpoint_uniform'),
             ('nafnet_bridge_uniform@t', 'nafnet_endpoint_uniform@t1'),
             ('nafnet_bridge_uniform@t', 'redcnn_endpoint_uniform'),
             ('redcnn_wide_bridge_uniform', 'redcnn_wide_endpoint_uniform'),
             ('redcnn_wide_t_bridge_uniform@t', 'redcnn_wide_endpoint_uniform'),
             ('redcnn_bridge_uniform_ep200', 'redcnn_endpoint_uniform_ep1000'),
             ('redcnn_bridge_uniform_ep300', 'redcnn_endpoint_uniform_ep1000'),
             ('redcnn_bridge_uniform_ep300', 'redcnn_wide_endpoint_uniform'),
             ('redcnn_bridge_equal_improvement_ep200', 'redcnn_endpoint_uniform_ep1000'),
             ('redcnn_bridge_equal_improvement_ep200', 'endpoint_redcnn'),
             ('redcnn_bridge_equal_improvement_ep200', 'redcnn_endpoint_equal_improvement_ep1000'),
             ('redcnn_bridge_uniform_ep200', 'redcnn_endpoint_uniform'),
             ('redcnn_wide_bridge_uniform_ep200', 'redcnn_wide_endpoint_uniform_ep1000'),
             ('redcnn_wide_bridge_uniform_ep200', 'redcnn_wide_endpoint_uniform'),
             ('redcnn_wide_t_bridge_equal_improvement@t', 'redcnn_wide_endpoint_uniform'),
             ('unet_res_bridge_uniform@t', 'unet_res_interp_uniform@t'),
             ('unet_res_bridge_uniform@t', 'unet_res_rep5@t1')]
    OURS = ['unet_res_bridge_uniform_aug_ema_ep160@t', 'unet_res_bridge_uniform_aug_ema_ep320@t',
            'unet_res_bridge_uniform_aug_ema_lr0.0002_ep160@t', 'unet_res96_bridge_uniform_aug_ema_ep160@t',
            'unet_res_bridge_uniform_aug_ema_bs32_ep160@t', 'rep4_unet_res_bridge_uniform_aug_ema@t',
            'rep4_unet_res_bridge_uniform_aug_ema_lr0.0002_ep80@t', 'rep4_unet_res96_bridge_uniform_aug_ema_lr0.0002_ep80@t',
            'unet_res96_bridge_uniform_aug_ema_lr0.0002_ep160@t']
    BASE = ['redcnn_endpoint_uniform_aug_ema_ep1000', 'redcnn_wide_endpoint_uniform_aug_ema_ep1000',
            'unet_res_endpoint_uniform_aug_ema_ep800@t1', 'unet_res_endpoint_uniform_aug_ema@t1',
            'drunet_endpoint_uniform_aug_ema_ep1000@t1', 'dncnn_endpoint_uniform_aug_ema_ep1000',
            'nafnet_endpoint_uniform_aug_ema_lr0.001_ep1000@t1', 'edcnn_endpoint_uniform_aug_ema_ep1000']
    pairs += [(o, b) for o in OURS for b in BASE]
    for frac in ('0.05', '0.1', '0.25'):
        pairs += [(f'unet_res_bridge_uniform_frac{frac}@t', f'unet_res_endpoint_uniform_frac{frac}@t1'),
                  (f'unet_res_bridge_uniform_frac{frac}@t', f'redcnn_endpoint_uniform_frac{frac}'),
                  (f'redcnn_t_bridge_uniform_frac{frac}@t', f'redcnn_endpoint_uniform_frac{frac}'),
                  (f'unet_res_bridge_uniform_frac{frac}@t', f'unet_res_interp_uniform_frac{frac}@t')]
    summary['wilcoxon'] = {}
    for a, b in pairs:
        if a not in per or b not in per:
            continue
        for key in data:
            if key in per[a] and key in per[b]:
                pa, pb = np.array(per[a][key]['psnr']), np.array(per[b][key]['psnr'])
                if len(pa) == len(pb) and len(pa) > 5 and np.any(pa != pb):
                    summary['wilcoxon'][f'{a}|{b}|{key}'] = {
                        'psnr_diff': float(np.mean(pa - pb)), 'psnr_p': float(wilcoxon(pa, pb).pvalue),
                        'win_rate': float(np.mean(pa > pb))}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    np.savez_compressed(os.path.join(out_dir, 'per_slice.npz'),
                        **{f'{m}|{k}|{met}': np.array(v) for m, byk in per.items()
                           for k, e in byk.items() for met, v in e.items() if v})

    keys = sorted(data.keys(), key=lambda k: knot_t[k])
    print("\nPSNR (dB) by knot (t / dose%):")
    print(f"{'model':42s}" + "".join(f"{knot_t[k]:.2f}/{100*(0.01 if ds=='2detect' else 0.1)**knot_t[k]:5.1f}% " for k in keys))
    for m in sorted(per):
        row = f"{m:42s}"
        for k in keys:
            row += f"{summary['models'][m][k]['psnr']:12.2f}" if k in summary['models'][m] else f"{'':12s}"
        print(row)
    print(f"\nwritten {out_dir}")


if __name__ == '__main__':
    main()
