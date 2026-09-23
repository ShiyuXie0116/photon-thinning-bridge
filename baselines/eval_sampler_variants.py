"""Comparison of reverse-step coefficients and multi-step sampler variants."""

import os, sys, json, argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified_eval import (UNetWithTime, compute_psnr, compute_ssim,
                          get_time_steps, get_test_slice_ids)
from baselines.train_baselines import REDCNN
from baselines.eval_all import try_lpips, lpips_val, BASES, ALPHAS
from baselines.train_bridge import UNetRes, REDCNNWide

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT_ROOT = f"{DATA_ROOT}/dose_bridge_baselines"


def coeff(sampler, t_next, t_curr, alpha):
    if t_next < 1e-8:
        return 0.0
    if sampler == 'linear':
        return t_next / t_curr
    r = (alpha ** (-t_next) - 1.0) / (alpha ** (-t_curr) - 1.0)
    return float(np.sqrt(r)) if sampler == 'std' else float(r)


def rollout(model, x1, times, sampler, alpha, device, blind=False):
    t_rev = times[::-1]
    x = x1.copy()
    states = []
    with torch.no_grad():
        for i in range(len(t_rev) - 1):
            t_curr, t_next = float(t_rev[i]), float(t_rev[i + 1])
            xt = torch.from_numpy(x).float()[None, None].to(device)
            if blind:
                out = model(xt).cpu().numpy().squeeze()
            else:
                tt = torch.tensor([t_curr], dtype=torch.float32).to(device)
                out = model(xt, tt).cpu().numpy().squeeze()
            x0p = np.clip(out, 0.0, 1.0)
            c = coeff(sampler, t_next, t_curr, alpha)
            x = np.clip(x0p + c * (x - x0p), 0.0, 1.0)
            states.append(x.copy())
    return states


def load_sim_pairs(dataset):
    base = BASES[dataset]
    ref = os.path.join(base, 'precomputed_equal_improvement')
    sids = sorted(get_test_slice_ids(dataset, ref))
    pairs = {}
    for sid in tqdm(sids, desc='load'):
        f = os.path.join(ref, f'{sid}_step4.npz')
        if os.path.exists(f):
            d = np.load(f)
            pairs[sid] = (d['x_t'].astype(np.float32), d['x0'].astype(np.float32))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True,
                    choices=['2detect', 'ldct', 'real_ldct', 'real_2detect'])
    ap.add_argument('--only', default='', help='comma list of model names to run; results are merged into the existing summary')
    args = ap.parse_args()
    device = torch.device('cuda')
    out_dir = os.path.join(OUT_ROOT, f'eval_sampler_{args.dataset}')
    os.makedirs(out_dir, exist_ok=True)

    # (name, ckpt, blind, schedule for knots, alpha)
    if args.dataset == '2detect':
        alpha = ALPHAS['2detect']
        base = BASES['2detect']
        models = [
            ('x0_uniform', f'{base}/output_x0pred/x0pred_best.pth', False, 'uniform', alpha),
            ('x0_EI', f'{base}/output_equal_improvement_x0pred/dose_bridge_best.pth',
             False, 'equal_improvement', alpha),
            ('redcnn_EI', f'{OUT_ROOT}/2detect_redcnn_bridge_equal_improvement/best.pth',
             True, 'equal_improvement', alpha),
            ('redcnn_EI_ep300', f'{OUT_ROOT}/2detect_redcnn_bridge_equal_improvement_ep300/best.pth',
             True, 'equal_improvement', alpha),
            ('wide_EI_ep200', f'{OUT_ROOT}/2detect_redcnn_wide_bridge_equal_improvement_ep200/best.pth',
             'wide', 'equal_improvement', alpha),
            ('wide_EI_ep300', f'{OUT_ROOT}/2detect_redcnn_wide_bridge_equal_improvement_ep300/best.pth',
             'wide', 'equal_improvement', alpha),
            ('ures_uniform', f'{OUT_ROOT}/2detect_unet_res_bridge_uniform/best.pth',
             'res', 'uniform', alpha),
            ('ures_EI', f'{OUT_ROOT}/2detect_unet_res_bridge_equal_improvement/best.pth',
             'res', 'equal_improvement', alpha),
            ('ures_aug160', f'{OUT_ROOT}/2detect_unet_res_bridge_uniform_aug_ema_ep160/best.pth',
             'res', 'uniform', alpha),
            ('ures_aug160_EI', f'{OUT_ROOT}/2detect_unet_res_bridge_uniform_aug_ema_ep160/best.pth',
             'res', 'equal_improvement', alpha),
            ('ures_aug320', f'{OUT_ROOT}/2detect_unet_res_bridge_uniform_aug_ema_ep320/best.pth',
             'res', 'uniform', alpha),
            ('ures_rep4', f'{OUT_ROOT}/2detect_rep4_unet_res_bridge_uniform_aug_ema/best.pth',
             'res', 'uniform', alpha),
            ('ures_rep4_160', f'{OUT_ROOT}/2detect_rep4_unet_res_bridge_uniform_aug_ema_ep160/best.pth',
             'res', 'uniform', alpha),
            ('ures96ref_rep4v2', f'{OUT_ROOT}/2detect_rep4_unet_res96_ref_bridge_uniform_aug_ema_lr0.0002_ep80v2/best.pth',
             'ref96', 'uniform', 0.01),
            ('ures96ref_rep4v2_EI', f'{OUT_ROOT}/2detect_rep4_unet_res96_ref_bridge_uniform_aug_ema_lr0.0002_ep80v2/best.pth',
             'ref96', 'equal_improvement', 0.01),
            ('ures96_rep4v2', f'{OUT_ROOT}/2detect_rep4_unet_res96_bridge_uniform_aug_ema_lr0.0002_ep80v2/best.pth',
             'res96', 'uniform', alpha),
            ('ures96_rep4v2_EI', f'{OUT_ROOT}/2detect_rep4_unet_res96_bridge_uniform_aug_ema_lr0.0002_ep80v2/best.pth',
             'res96', 'equal_improvement', alpha),
            ('hyb96_rep4', f'{OUT_ROOT}/2detect_rep4_hybrid96_bridge_uniform_aug_ema_lr0.0002_ep80/best.pth',
             'hyb96', 'uniform', alpha),
            ('hyb96_rep4_EI', f'{OUT_ROOT}/2detect_rep4_hybrid96_bridge_uniform_aug_ema_lr0.0002_ep80/best.pth',
             'hyb96', 'equal_improvement', alpha),
            ('ures96_aug160', f'{OUT_ROOT}/2detect_unet_res96_bridge_uniform_aug_ema_ep160/best.pth',
             'res96', 'uniform', alpha),
            ('ures48_aug160', f'{OUT_ROOT}/2detect_unet_res48_bridge_uniform_aug_ema_ep160/best.pth',
             'res48', 'uniform', alpha),
            ('ures_lr2e4', f'{OUT_ROOT}/2detect_unet_res_bridge_uniform_aug_ema_lr0.0002_ep160/best.pth',
             'res', 'uniform', alpha),
            ('ures_bs32', f'{OUT_ROOT}/2detect_unet_res_bridge_uniform_aug_ema_bs32_ep160/best.pth',
             'res', 'uniform', alpha),
        ]
        pairs = load_sim_pairs('2detect')
    elif args.dataset == 'ldct':
        alpha = ALPHAS['ldct']
        base = BASES['ldct']
        models = [
            ('x0_uniform', f'{base}/output_x0pred/ldct_dose_bridge_best.pth',
             False, 'uniform', alpha),
            ('x0_dose_unif', f'{base}/output_geometric_x0pred/ldct_dose_bridge_best.pth',
             False, 'geometric', alpha),
            ('x0_EI', f'{base}/output_equal_improvement_x0pred/ldct_dose_bridge_best.pth',
             False, 'equal_improvement', alpha),
            ('ures_uniform', f'{OUT_ROOT}/ldct_unet_res_bridge_uniform/best.pth',
             'res', 'uniform', alpha),
            ('ures_EI', f'{OUT_ROOT}/ldct_unet_res_bridge_equal_improvement/best.pth',
             'res', 'equal_improvement', alpha),
            ('redcnn_EI_ep200', f'{OUT_ROOT}/ldct_redcnn_bridge_equal_improvement_ep200/best.pth',
             True, 'equal_improvement', alpha),
            ('wide_EI_ep200', f'{OUT_ROOT}/ldct_redcnn_wide_bridge_equal_improvement_ep200/best.pth',
             'wide', 'equal_improvement', alpha),
            ('ures_aug', f'{OUT_ROOT}/ldct_unet_res_bridge_uniform_aug_ema_ep160/best.pth',
             'res', 'uniform', alpha),
                ('ldhyb4_real', f'{OUT_ROOT}/ldct_rep4_hybrid_bridge_uniform_aug_ema_lr0.0002_ep21/best.pth',
                 'hyb', 'uniform', alpha),
                ('ldhyb1_real', f'{OUT_ROOT}/ldct_hybrid_bridge_uniform_aug_ema_lr0.0002_ep85/best.pth',
                 'hyb', 'uniform', alpha),
            ('ures_aug_lr2e4', f'{OUT_ROOT}/ldct_unet_res_bridge_uniform_aug_ema_lr0.0002_ep160/best.pth',
             'res', 'uniform', alpha),
            ('ures96_aug_lr2e4', f'{OUT_ROOT}/ldct_unet_res96_bridge_uniform_aug_ema_lr0.0002_ep160/best.pth',
             'res96', 'uniform', alpha),
        ]
        pairs = load_sim_pairs('ldct')
    else:
        import eval_real_lowdose as ER
        from baselines import eval_real_all as ERA
        if args.dataset == 'real_ldct':
            alpha = 0.1
            models = [
                ('x0_dose_unif',
                 f"{DATA_ROOT}/LDCT_dose_bridge/output_geometric_x0pred/ldct_dose_bridge_best.pth",
                 False, 'geometric', alpha),
                ('redcnn_uniform', f'{OUT_ROOT}/ldct_redcnn_bridge_uniform/best.pth',
                 True, 'geometric', alpha),
                ('ures_uniform', f'{OUT_ROOT}/ldct_unet_res_bridge_uniform/best.pth',
                 'res', 'uniform', alpha),
                ('redcnn_EI_ep200', f'{OUT_ROOT}/ldct_redcnn_bridge_equal_improvement_ep200/best.pth',
                 True, 'equal_improvement', alpha),
                ('wide_EI_ep200', f'{OUT_ROOT}/ldct_redcnn_wide_bridge_equal_improvement_ep200/best.pth',
                 'wide', 'equal_improvement', alpha),
                ('ures_aug', f'{OUT_ROOT}/ldct_unet_res_bridge_uniform_aug_ema_ep160/best.pth',
                 'res', 'uniform', alpha),
                ('ures_aug_lr2e4', f'{OUT_ROOT}/ldct_unet_res_bridge_uniform_aug_ema_lr0.0002_ep160/best.pth',
                 'res', 'uniform', alpha),
                ('ures96_aug_lr2e4', f'{OUT_ROOT}/ldct_unet_res96_bridge_uniform_aug_ema_lr0.0002_ep160/best.pth',
                 'res96', 'uniform', alpha),
                ('ldhyb4_real', f'{OUT_ROOT}/ldct_rep4_hybrid_bridge_uniform_aug_ema_lr0.0002_ep21/best.pth',
                 'hyb', 'uniform', alpha),
                ('ldhyb1_real', f'{OUT_ROOT}/ldct_hybrid_bridge_uniform_aug_ema_lr0.0002_ep85/best.pth',
                 'hyb', 'uniform', alpha),
            ]
            pairs = ERA.cache_ldct()
        else:
            a_eff = 425.0 / 15000.0
            a_real = 1500.0 / 53000.0
            models = [
                ('x0_effI0',
                 f"{DATA_ROOT}/2DeteCT_dose_bridge_effI0/output_geometric_x0pred/dose_bridge_best.pth",
                 False, 'geometric', a_eff),
                ('x0_realI0',
                 f"{DATA_ROOT}/2DeteCT_dose_bridge_realI0/output_geometric_x0pred/dose_bridge_best.pth",
                 False, 'geometric', a_real),
                ('ures_realI0', f'{OUT_ROOT}/2detect_realI0_unet_res_bridge_geometric/best.pth',
                 'res', 'geometric', a_real),
                ('ures_effI0', f'{OUT_ROOT}/2detect_effI0_unet_res_bridge_geometric/best.pth',
                 'res', 'geometric', a_eff),
                ('redcnn_effI0_ep200', f'{OUT_ROOT}/2detect_effI0_redcnn_bridge_geometric_ep200/best.pth',
                 True, 'geometric', a_eff),
                ('wide_effI0_ep200', f'{OUT_ROOT}/2detect_effI0_redcnn_wide_bridge_geometric_ep200/best.pth',
                 'wide', 'geometric', a_eff),
                ('ures_effI0_aug', f'{OUT_ROOT}/2detect_effI0_unet_res_bridge_geometric_aug_ema_ep160/best.pth',
                 'res', 'geometric', a_eff),
                ('ures96_effI0_aug', f'{OUT_ROOT}/2detect_effI0_unet_res96_bridge_geometric_aug_ema_lr0.0002_ep160/best.pth',
                 'res96', 'geometric', a_eff),
                ('ures96ref_effI0_aug', f'{OUT_ROOT}/2detect_effI0_unet_res96_ref_bridge_geometric_aug_ema_lr0.0002_ep160/best.pth',
                 'ref96', 'geometric', a_eff),
                ('hyb96_effI0_aug', f'{OUT_ROOT}/2detect_effI0_hybrid96_bridge_geometric_aug_ema_lr0.0002_ep160/best.pth',
                 'hyb96', 'geometric', a_eff),
            ]
            alpha = a_real
            pairs = ERA.cache_2detect()

    sids = sorted(pairs.keys())
    print(f"{args.dataset}: {len(sids)} slices")
    lp = try_lpips(device)
    per = {}

    def record(method, pred, gt):
        e = per.setdefault(method, {'psnr': [], 'ssim': [], 'lpips': []})
        e['psnr'].append(compute_psnr(pred, gt))
        e['ssim'].append(compute_ssim(pred, gt))
        v = lpips_val(lp, pred, gt, device)
        if v is not None:
            e['lpips'].append(v)

    only = [m for m in args.only.split(',') if m]
    for name, ckpt, blind, schedule, a in models:
        if only and name not in only:
            continue
        if not os.path.exists(ckpt):
            print(f"SKIP {name}: {ckpt}")
            continue
        if blind in ('res', 'res96', 'res48'):
            model = UNetRes(base_ch={'res': 64, 'res96': 96, 'res48': 48}[blind]).to(device)
            blind = False
        elif blind in ('ref', 'refw', 'ref96', 'hyb', 'hyb96', 'wfilm'):
            from baselines.train_bridge import build as _build_bridge
            model = _build_bridge({'ref': 'unet_res_ref', 'refw': 'unet_res_ref_w', 'ref96': 'unet_res96_ref',
                               'hyb': 'hybrid', 'hyb96': 'hybrid96', 'wfilm': 'wide_film'}[blind])[0].to(device)
            blind = False
        elif blind == 'wide':
            model = REDCNNWide().to(device)
            blind = True
        else:
            model = (REDCNN() if blind else UNetWithTime(base_ch=64, t_dim=128)).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device)['model'])
        model.eval()
        configs = [(s, 5) for s in ('linear', 'std', 'var')]
        configs += [('var', 10)]
        if not blind:
            configs += [('linear', 10)]  # continuous-t extrapolation
        for sampler, N in configs:
            times = get_time_steps(a, N, schedule)
            tag = f'{name}_{sampler}_N{N}'
            for sid in tqdm(sids, desc=tag):
                x1, x0 = pairs[sid]
                states = rollout(model, x1, times, sampler, a, device, blind)
                record(f'{tag}_final', states[-1], x0)
                record(f'{tag}_pen', states[-2], x0)
        del model
        torch.cuda.empty_cache()

    ps_path = os.path.join(out_dir, 'per_slice.npz')
    arrays = {f'{m}__{k}': np.array(v[k]) for m, v in per.items() for k in v if v[k]}
    if only and os.path.exists(ps_path):
        old = np.load(ps_path)
        if list(old['slice_ids']) == list(sids):
            for k in old.files:
                if k != 'slice_ids' and k not in arrays:
                    arrays[k] = old[k]
    np.savez_compressed(ps_path, slice_ids=np.array(sids), **arrays)

    from scipy.stats import wilcoxon
    summary, pv = {}, {}
    if only and os.path.exists(os.path.join(out_dir, 'summary.json')):
        prev = json.load(open(os.path.join(out_dir, 'summary.json')))
        summary.update(prev.get('summary', {})); pv.update(prev.get('wilcoxon_vs_linear', {}))
    for m, v in sorted(per.items()):
        summary[m] = {'psnr_mean': float(np.mean(v['psnr'])),
                      'psnr_std': float(np.std(v['psnr'])),
                      'ssim_mean': float(np.mean(v['ssim'])),
                      'lpips_mean': float(np.mean(v['lpips'])) if v['lpips'] else None,
                      'n': len(v['psnr'])}
    for m in per:
        if '_linear_' in m:
            continue
        base_m = m
        for s in ('_std_', '_var_'):
            base_m = base_m.replace(s, '_linear_')
        if base_m in per and len(per[m]['psnr']) == len(per[base_m]['psnr']):
            entry = {}
            for metric in ('psnr', 'ssim', 'lpips'):
                a1, b1 = np.array(per[m][metric]), np.array(per[base_m][metric])
                if len(a1) and len(a1) == len(b1):
                    try:
                        entry[f'{metric}_p'] = float(wilcoxon(a1, b1).pvalue)
                        entry[f'{metric}_diff'] = float(np.mean(a1) - np.mean(b1))
                    except Exception:
                        pass
            pv[f'{m}|{base_m}'] = entry
    result = {'dataset': args.dataset, 'summary': summary, 'wilcoxon_vs_linear': pv}
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nSAMPLER STUDY — {args.dataset}")
    for m in sorted(summary):
        s = summary[m]
        lpv = f"{s['lpips_mean']:.4f}" if s['lpips_mean'] is not None else '---'
        print(f"{m:<34s}{s['psnr_mean']:>8.2f}±{s['psnr_std']:<5.2f}"
              f"{s['ssim_mean']:>8.4f}{lpv:>9s}")
    print(f"saved -> {out_dir}")


if __name__ == '__main__':
    main()
