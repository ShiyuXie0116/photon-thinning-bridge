"""Extended image-quality metrics on the real low-dose scans."""
import os, sys, json, argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from baselines import eval_real_all as ERA
from baselines.train_baselines import REDCNN
from baselines.train_bridge import UNetRes, build as build_bridge
from baselines.eval_sampler_variants import rollout
from unified_eval import compute_psnr, compute_ssim, get_time_steps

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT_ROOT = f"{DATA_ROOT}/dose_bridge_baselines"
HIGHER = {'psnr', 'ssim', 'msssim', 'fsim', 'vif', 'haarpsi'}
LOWER = {'rmse', 'mae', 'gmsd', 'dists', 'lpips_alex', 'lpips_vgg'}
ORDER = ['psnr', 'ssim', 'rmse', 'mae', 'msssim', 'fsim', 'vif', 'gmsd', 'haarpsi', 'dists', 'lpips_alex', 'lpips_vgg']


def table_methods(ds):
    O = OUT_ROOT
    if ds == 'ldct':
        E = 'ldct_{}_endpoint_equal_improvement_ep100'
        rows = [('DnCNN', f'{O}/{E.format("dncnn")}/best.pth', 'r6:dncnn'),
                ('EDCNN', f'{O}/{E.format("edcnn")}/best.pth', 'r6:edcnn'),
                ('NAFNet', f'{O}/{E.format("nafnet")}/best.pth', 'r6:nafnet'),
                ('DRUNet', f'{O}/{E.format("drunet")}/best.pth', 'r6:drunet'),
                ('MAP-NN', f'{O}/{E.format("mapnn")}/best.pth', 'r6:mapnn'),
                ('Restormer', f'{O}/{E.format("restormer")}/best.pth', 'r6:restormer'),
                ('WGAN-VGG', f'{O}/ldct_wganvgg_endpoint_equal_improvement_pub/best.pth', 'r6:wganvgg'),
                ('U-Net, single dose', f'{O}/{E.format("unet_res")}/best.pth', 'unet_res_ep'),
                ('RED-CNN', f'{O}/{E.format("redcnn")}/best.pth', 'redcnn_endpoint')]
        ours = dict(ckpt=f'{O}/ldct_rep4_hybrid_bridge_uniform_aug_ema_lr0.0002_ep21/best.pth', arch='hybrid',
                    schedule='uniform', alpha=0.1, t_hat=f'{O}/eval_real_ldct_tsweep/t_hat_ldct.json',
                    pid_of=lambda sid: sid.split('_s')[0])
    else:
        E = '2detect_effI0_{}_endpoint_geometric_ep100'
        rows = [('DnCNN', f'{O}/{E.format("dncnn")}/best.pth', 'r6:dncnn'),
                ('EDCNN', f'{O}/{E.format("edcnn")}/best.pth', 'r6:edcnn'),
                ('NAFNet', f'{O}/{E.format("nafnet")}/best.pth', 'r6:nafnet'),
                ('DRUNet', f'{O}/{E.format("drunet")}/best.pth', 'r6:drunet'),
                ('MAP-NN', f'{O}/{E.format("mapnn")}/best.pth', 'r6:mapnn'),
                ('Restormer', f'{O}/{E.format("restormer")}/best.pth', 'r6:restormer'),
                ('WGAN-VGG', f'{O}/2detect_effI0_wganvgg_endpoint_geometric_pub/best.pth', 'r6:wganvgg'),
                ('U-Net, single dose', f'{O}/{E.format("unet_res")}/best.pth', 'unet_res_ep'),
                ('RED-CNN', f'{O}/{E.format("redcnn")}/best.pth', 'redcnn_endpoint')]
        ours = dict(ckpt=f'{O}/2detect_effI0_hybrid96_bridge_geometric_aug_ema_lr0.0002_ep160/best.pth', arch='hybrid96',
                    schedule='geometric', alpha=425.0 / 15000.0, t_hat=f'{O}/eval_real_ldct_tsweep/t_hat_2detect.json',
                    pid_of=lambda sid: 'all')
    return rows, ours


def load_model(path, kind, device):
    sd = torch.load(path, map_location=device)
    timed = False
    if kind == 'redcnn_endpoint':
        model = REDCNN()
    elif kind == 'unet_res_ep':
        model, timed = UNetRes(base_ch=64), True   # UNetRes.forward(x, t); endpoint nets are queried at t=1
    elif kind.startswith('r6:'):
        model, timed = build_bridge(kind[3:])
    else:
        raise ValueError(kind)
    model = model.to(device)
    model.load_state_dict(sd['model'])
    model.eval()
    return model, timed


class Scorer:
    def __init__(self, device):
        self.device = device
        import piq
        self.piq = piq
        self.lp_alex = self.lp_vgg = self.dists = None
        try:
            import lpips
            self.lp_alex = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
        except Exception as e:
            print(f'LPIPS(alex) unavailable: {e}')
        try:
            import lpips
            self.lp_vgg = lpips.LPIPS(net='vgg', verbose=False).to(device).eval()
        except Exception as e:
            print(f'LPIPS(vgg) unavailable: {e}')
        try:
            self.dists = piq.DISTS(reduction='none').to(device).eval()
        except Exception as e:
            print(f'DISTS unavailable: {e}')

    def __call__(self, pred, gt):
        pred = np.clip(np.asarray(pred, np.float32), 0, 1); gt = np.clip(np.asarray(gt, np.float32), 0, 1)
        d = {'psnr': float(compute_psnr(pred, gt)), 'ssim': float(compute_ssim(pred, gt))}
        err = pred - gt
        d['rmse'] = float(np.sqrt(np.mean(err ** 2))); d['mae'] = float(np.mean(np.abs(err)))
        piq = self.piq
        with torch.no_grad():
            x = torch.from_numpy(pred)[None, None].to(self.device); y = torch.from_numpy(gt)[None, None].to(self.device)
            x3, y3 = x.repeat(1, 3, 1, 1), y.repeat(1, 3, 1, 1)
            d['msssim'] = float(piq.multi_scale_ssim(x, y, data_range=1.0))
            d['fsim'] = float(piq.fsim(x3, y3, data_range=1.0, chromatic=False))
            d['vif'] = float(piq.vif_p(x, y, data_range=1.0))
            d['gmsd'] = float(piq.gmsd(x, y, data_range=1.0))
            d['haarpsi'] = float(piq.haarpsi(x, y, data_range=1.0))
            if self.dists is not None:
                d['dists'] = float(self.dists(x3, y3).squeeze())
            if self.lp_alex is not None:
                d['lpips_alex'] = float(self.lp_alex(x3 * 2 - 1, y3 * 2 - 1).item())
            if self.lp_vgg is not None:
                d['lpips_vgg'] = float(self.lp_vgg(x3 * 2 - 1, y3 * 2 - 1).item())
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['ldct', '2detect'])
    ap.add_argument('--no_save_pred', action='store_true')
    ap.add_argument('--limit', type=int, default=0, help='debug: only the first N slices')
    args = ap.parse_args()
    device = torch.device('cuda')
    out_dir = os.path.join(OUT_ROOT, f'eval_real_metrics_{args.dataset}'); os.makedirs(out_dir, exist_ok=True)
    pairs = ERA.cache_ldct() if args.dataset == 'ldct' else ERA.cache_2detect()
    sids = sorted(pairs)
    if args.limit:
        sids = sids[:args.limit]
    print(f'{args.dataset}: {len(sids)} real test slices', flush=True)
    rows, ours = table_methods(args.dataset)
    score = Scorer(device)
    per, preds = {}, {}

    def rec(method, sid, pred, gt):
        d = score(pred, gt)
        e = per.setdefault(method, {})
        for k, v in d.items():
            e.setdefault(k, []).append(v)
        if not args.no_save_pred:
            preds[f'{method}|{sid}'] = np.asarray(pred, np.float32)

    for sid in tqdm(sids, desc='Low-dose input'):
        x1, x0 = pairs[sid]; rec('Low-dose input', sid, x1, x0)

    for label, path, kind in rows:
        if not os.path.exists(path):
            print(f'SKIP {label}: {path}'); continue
        model, timed = load_model(path, kind, device)
        for sid in tqdm(sids, desc=label):
            x1, x0 = pairs[sid]
            with torch.no_grad():
                xt = torch.from_numpy(x1).float()[None, None].to(device)
                out = model(xt, torch.tensor([1.0], device=device)) if timed else model(xt)
            rec(label, sid, np.clip(out.cpu().numpy().squeeze(), 0, 1), x0)
        del model; torch.cuda.empty_cache()

    model = build_bridge(ours['arch'])[0].to(device)
    model.load_state_dict(torch.load(ours['ckpt'], map_location=device)['model']); model.eval()
    a, sched = ours['alpha'], ours['schedule']
    TH = json.load(open(ours['t_hat'])) if os.path.exists(ours['t_hat']) else None
    if TH is not None:
        print('reference-free t_hat:', {k: round(v['t_hat_median'], 3) for k, v in TH['per_patient'].items()}, flush=True)
    times1 = get_time_steps(a, 5, sched)

    def x0hat(x, t):
        with torch.no_grad():
            xt = torch.from_numpy(x).float()[None, None].to(device)
            return np.clip(model(xt, torch.tensor([float(t)], device=device)).cpu().numpy().squeeze(), 0, 1)

    def pm_traj(x1, times):
        """5-step posterior-mean trajectory ('var' coefficients) from times[-1] down to 0;
        returns (final state, mean of the x0-predictions)."""
        t_rev = times[::-1]; x = x1.copy(); preds_ = []
        for i in range(len(t_rev) - 1):
            tc, tn = float(t_rev[i]), float(t_rev[i + 1])
            p = x0hat(x, tc); preds_.append(p)
            r = 0.0 if tn < 1e-8 else (a ** (-tn) - 1.0) / (a ** (-tc) - 1.0)
            x = np.clip(p + r * (x - p), 0, 1)
        return x, np.clip(np.mean(preds_, 0), 0, 1)

    timesEI = get_time_steps(a, 5, 'equal_improvement')
    for sid in tqdm(sids, desc='Ours'):
        x1, x0 = pairs[sid]
        rec('Ours, N=1', sid, x0hat(x1, 1.0), x0)
        states = rollout(model, x1, times1, 'var', a, device, False)   # == Table-3 "Ours" (eval_sampler_variants, final state)
        rec('Ours', sid, states[-1], x0)
        xf, pa = pm_traj(x1, times1); rec(f'Ours ({sched}, pm-avg)', sid, pa, x0)
        xf, pa = pm_traj(x1, timesEI); rec('Ours (EI, final)', sid, xf, x0); rec('Ours (EI, pm-avg)', sid, pa, x0)
        if TH is not None:
            th = float(TH['per_patient'][ours['pid_of'](sid)]['t_hat_median'])
            rec('Ours, N=1 at t_hat', sid, x0hat(x1, th), x0)
            xf, pa = pm_traj(x1, times1 * th); rec(f'Ours at t_hat ({sched}, final)', sid, xf, x0); rec(f'Ours at t_hat ({sched}, pm-avg)', sid, pa, x0)
            xf, pa = pm_traj(x1, timesEI * th); rec('Ours at t_hat (EI, final)', sid, xf, x0); rec('Ours at t_hat (EI, pm-avg)', sid, pa, x0)
    del model; torch.cuda.empty_cache()

    if preds:
        np.savez_compressed(os.path.join(out_dir, 'preds_t3.npz'), **preds)
    np.savez_compressed(os.path.join(out_dir, 'per_slice.npz'), slice_ids=np.array(sids),
                        **{f'{m}__{k}': np.array(v) for m, e in per.items() for k, v in e.items()})
    summary = {m: {k: {'mean': float(np.mean(v)), 'std': float(np.std(v)), 'n': len(v)} for k, v in e.items()} for m, e in per.items()}
    metrics = [k for k in ORDER if any(k in e for e in per.values())]
    best = {}
    for k in metrics:
        vals = {m: summary[m][k]['mean'] for m in summary if k in summary[m] and not m.startswith('Low-dose') and (not m.startswith('Ours') or m == 'Ours')}
        best[k] = (max if k in HIGHER else min)(vals, key=vals.get)
    json.dump({'dataset': args.dataset, 'n': len(sids), 'metrics': metrics, 'best': best, 'summary': summary,
               'ours': {k: v for k, v in ours.items() if k != 'pid_of'}},
              open(os.path.join(out_dir, 'summary.json'), 'w'), indent=1)
    hdr = f"{'method':34s}" + ''.join(f'{k:>10s}' for k in metrics)
    print('\n' + hdr)
    md = ['| method | ' + ' | '.join(f"{k}{'↑' if k in HIGHER else '↓'}" for k in metrics) + ' |', '|' + '---|' * (len(metrics) + 1)]
    for m in per:
        cells = []
        for k in metrics:
            if k in summary[m]:
                v = summary[m][k]['mean']; s = f'{v:.2f}' if k == 'psnr' else f'{v:.4f}'
                cells.append(('**' + s + '**') if best.get(k) == m else s)
            else:
                cells.append('--')
        print(f'{m:34s}' + ''.join(f'{c.replace("**", ""):>10s}' for c in cells))
        md.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    open(os.path.join(out_dir, 'table.md'), 'w').write('\n'.join(md) + '\n')
    print(f'\nsaved -> {out_dir}')


if __name__ == '__main__':
    main()
