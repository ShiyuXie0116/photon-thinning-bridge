"""Evaluation at the training dose on the simulated test sets."""

import os, sys, json, argparse
import numpy as np
import torch
from glob import glob
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified_eval import (UNetWithTime, compute_psnr, compute_ssim,
                          get_time_steps, get_test_slice_ids)
from baselines.train_baselines import REDCNN, UNetEndpoint
from baselines.train_boosters import REDCNNTime

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT_ROOT = f"{DATA_ROOT}/dose_bridge_baselines"

BASES = {
    '2detect': f"{DATA_ROOT}/2DeteCT_dose_bridge",
    'ldct': f"{DATA_ROOT}/LDCT_dose_bridge",
}
ALPHAS = {'2detect': 0.01, 'ldct': 0.1}


def try_lpips(device):
    try:
        import lpips
        net = lpips.LPIPS(net='alex').to(device)
        net.eval()
        return net
    except Exception as e:
        print(f"LPIPS unavailable ({e}); skipping perceptual metric")
        return None


def lpips_val(lp, pred, gt, device):
    if lp is None:
        return None
    with torch.no_grad():
        a = torch.from_numpy(pred).float()[None, None].repeat(1, 3, 1, 1).to(device) * 2 - 1
        b = torch.from_numpy(gt).float()[None, None].repeat(1, 3, 1, 1).to(device) * 2 - 1
        return float(lp(a, b).item())


def run_trajectory(model, x_lowdose, schedule_times, pred_target, device):
    """Multi-step inference t=1 -> t=0; returns list of states after each step."""
    t_rev = schedule_times[::-1]
    x = x_lowdose.copy()
    states = []
    with torch.no_grad():
        for i in range(len(t_rev) - 1):
            t_curr, t_next = float(t_rev[i]), float(t_rev[i + 1])
            xt = torch.from_numpy(x).float()[None, None].to(device)
            tt = torch.tensor([t_curr], dtype=torch.float32).to(device)
            out = model(xt, tt).cpu().numpy().squeeze()
            if pred_target == 'x0':
                x0p = np.clip(out, 0.0, 1.0)
                x = x0p if t_next < 1e-6 else x0p + (t_next / t_curr) * (x - x0p)
            else:
                x = x + out * (t_curr - t_next)
            x = np.clip(x, 0.0, 1.0)
            states.append(x.copy())
    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['2detect', 'ldct'])
    args = ap.parse_args()

    device = torch.device('cuda')
    base = BASES[args.dataset]
    alpha = ALPHAS[args.dataset]
    n_steps = 5
    ref = os.path.join(base, 'precomputed_equal_improvement')
    out_dir = os.path.join(OUT_ROOT, f'eval_{args.dataset}')
    os.makedirs(out_dir, exist_ok=True)

    ckpt_name = 'dose_bridge_best.pth' if args.dataset == '2detect' else 'ldct_dose_bridge_best.pth'
    x0_uniform_ckpt = (f'{base}/output_x0pred/x0pred_best.pth' if args.dataset == '2detect'
                       else f'{base}/output_x0pred/{ckpt_name}')  # new LDCT x0-uniform run
    bridge_models = [
        ('vel_uniform', f'{base}/output/{ckpt_name}', 'velocity', 'uniform'),
        ('vel_dose_unif', f'{base}/output_geometric/{ckpt_name}', 'velocity', 'geometric'),
        ('vel_EI', f'{base}/output_equal_improvement/{ckpt_name}', 'velocity', 'equal_improvement'),
        ('x0_uniform', x0_uniform_ckpt, 'x0', 'uniform'),
        ('x0_dose_unif', f'{base}/output_geometric_x0pred/{ckpt_name}', 'x0', 'geometric'),
        ('x0_EI', f'{base}/output_equal_improvement_x0pred/{ckpt_name}', 'x0', 'equal_improvement'),
    ]
    baseline_models = [
        ('endpoint_unet', f'{OUT_ROOT}/{args.dataset}_unet_endpoint/best.pth', 'unet'),
        ('endpoint_redcnn', f'{OUT_ROOT}/{args.dataset}_redcnn_endpoint/best.pth', 'redcnn'),
    ]
    booster_models = [
        ('bridge_redcnn_uniform',
         f'{OUT_ROOT}/{args.dataset}_redcnn_bridge_uniform/best.pth', 'redcnn', 'uniform'),
        ('bridge_redcnn_EI',
         f'{OUT_ROOT}/{args.dataset}_redcnn_bridge_equal_improvement/best.pth',
         'redcnn', 'equal_improvement'),
        ('interp_unet',
         f'{OUT_ROOT}/{args.dataset}_interp_unet/best.pth', 'unet_t', 'uniform'),
        ('bridge_redcnn_t_EI',
         f'{OUT_ROOT}/{args.dataset}_redcnn_t_bridge_equal_improvement/best.pth',
         'redcnn_t', 'equal_improvement'),
        ('bridge_redcnn_t_uniform',
         f'{OUT_ROOT}/{args.dataset}_redcnn_t_bridge_uniform/best.pth',
         'redcnn_t', 'uniform'),
    ]
    # seed replicates (single-step / single-pass only)
    seed_models = ([(f'x0_uniform_seed{s}',
                     f'{OUT_ROOT}/{args.dataset}_unet_bridge_uniform_seed{s}/best.pth',
                     'unet_t') for s in (123, 2024)]
                   + [(f'endpoint_unet_seed{s}',
                       f'{OUT_ROOT}/{args.dataset}_unet_endpoint_seed{s}/best.pth',
                       'unet_endpoint') for s in (123, 2024)]
                   + [('endpoint_redcnn_t',
                       f'{OUT_ROOT}/{args.dataset}_redcnn_t_endpoint/best.pth',
                       'redcnn_t1')])

    test_ids = get_test_slice_ids(args.dataset, ref)
    test_ids = sorted(test_ids)
    print(f"{args.dataset}: {len(test_ids)} test slices, alpha={alpha}, ref={ref}")
    data = {}
    for sid in tqdm(test_ids, desc='load'):
        f = os.path.join(ref, f'{sid}_step4.npz')
        if os.path.exists(f):
            d = np.load(f)
            data[sid] = (d['x_t'].astype(np.float32), d['x0'].astype(np.float32))
    sids = sorted(data.keys())
    print(f"loaded {len(sids)}")

    lp = try_lpips(device)
    per = {}
    meta = {}
    singles = {}

    def stash(method, sid, pred):
        singles.setdefault(method, {})[sid] = pred.astype(np.float32)

    def record(method, pred, gt):
        e = per.setdefault(method, {'psnr': [], 'ssim': [], 'lpips': []})
        e['psnr'].append(compute_psnr(pred, gt))
        e['ssim'].append(compute_ssim(pred, gt))
        v = lpips_val(lp, pred, gt, device)
        if v is not None:
            e['lpips'].append(v)

    for sid in sids:
        x1, x0 = data[sid]
        record('low_dose', x1, x0)

    for name, ckpt_path, target, schedule in bridge_models:
        if not os.path.exists(ckpt_path):
            print(f"SKIP {name}: no ckpt {ckpt_path}")
            continue
        model = UNetWithTime(base_ch=64, t_dim=128).to(device)
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model'])
        model.eval()
        meta[name] = {'epoch': int(ck.get('epoch', -1)), 'target': target,
                      'schedule': schedule}
        times = get_time_steps(alpha, n_steps, schedule)
        for sid in tqdm(sids, desc=name):
            x1, x0 = data[sid]
            states = run_trajectory(model, x1, times, target, device)
            for k, xs in enumerate(states):
                record(f'{name}_step{k+1}', xs, x0)
            if target == 'x0':
                with torch.no_grad():
                    xt = torch.from_numpy(x1).float()[None, None].to(device)
                    tt = torch.tensor([1.0], dtype=torch.float32).to(device)
                    x0p = np.clip(model(xt, tt).cpu().numpy().squeeze(), 0, 1)
                record(f'{name}_single', x0p, x0)
                stash(f'{name}_single', sid, x0p)
        del model
        torch.cuda.empty_cache()

    for name, ckpt_path, arch in baseline_models:
        if not os.path.exists(ckpt_path):
            print(f"SKIP {name}: no ckpt {ckpt_path}")
            continue
        model = (UNetEndpoint() if arch == 'unet' else REDCNN()).to(device)
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model'])
        model.eval()
        meta[name] = {'epoch': int(ck.get('epoch', -1)), 'target': 'endpoint',
                      'schedule': '-'}
        for sid in tqdm(sids, desc=name):
            x1, x0 = data[sid]
            with torch.no_grad():
                pred = model(torch.from_numpy(x1).float()[None, None].to(device))
            pred = np.clip(pred.cpu().numpy().squeeze(), 0, 1)
            record(name, pred, x0)
            stash(name, sid, pred)
        del model
        torch.cuda.empty_cache()

    class BlindWrap(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x, t=None):
            return self.net(x)

    for name, ckpt_path, arch, schedule in booster_models:
        if not os.path.exists(ckpt_path):
            print(f"SKIP {name}: no ckpt {ckpt_path}")
            continue
        try:
            if arch == 'redcnn':
                model = BlindWrap(REDCNN()).to(device)
                sd = torch.load(ckpt_path, map_location=device)
                model.net.load_state_dict(sd['model'])
            elif arch == 'redcnn_t':
                model = REDCNNTime().to(device)
                sd = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(sd['model'])
            else:
                model = UNetWithTime(base_ch=64, t_dim=128).to(device)
                sd = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(sd['model'])
        except Exception as e:
            print(f"SKIP {name}: load failed ({e})")
            continue
        model.eval()
        meta[name] = {'epoch': int(sd.get('epoch', -1)), 'target': 'x0',
                      'schedule': schedule, 'arch': arch}
        times = get_time_steps(alpha, n_steps, schedule)
        for sid in tqdm(sids, desc=name):
            x1, x0 = data[sid]
            states = run_trajectory(model, x1, times, 'x0', device)
            for k, xs in enumerate(states):
                record(f'{name}_step{k+1}', xs, x0)
            with torch.no_grad():
                xt = torch.from_numpy(x1).float()[None, None].to(device)
                tt = torch.tensor([1.0], dtype=torch.float32).to(device)
                x0p = np.clip(model(xt, tt).cpu().numpy().squeeze(), 0, 1)
            record(f'{name}_single', x0p, x0)
            stash(f'{name}_single', sid, x0p)
        del model
        torch.cuda.empty_cache()

    for name, ckpt_path, arch in seed_models:
        if not os.path.exists(ckpt_path):
            print(f"SKIP {name}: no ckpt {ckpt_path}")
            continue
        try:
            if arch == 'unet_t':
                model = UNetWithTime(base_ch=64, t_dim=128).to(device)
            elif arch == 'redcnn_t1':
                model = REDCNNTime().to(device)
            else:
                model = UNetEndpoint().to(device)
            sd = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(sd['model'])
        except Exception as e:
            print(f"SKIP {name}: load failed ({e})")
            continue
        model.eval()
        meta[name] = {'epoch': int(sd.get('epoch', -1)), 'seed': int(sd.get('seed', -1))}
        for sid in tqdm(sids, desc=name):
            x1, x0 = data[sid]
            with torch.no_grad():
                xt = torch.from_numpy(x1).float()[None, None].to(device)
                if arch in ('unet_t', 'redcnn_t1'):
                    tt = torch.tensor([1.0], dtype=torch.float32).to(device)
                    pred = model(xt, tt)
                else:
                    pred = model(xt)
            suffix = '_single' if arch == 'unet_t' else ''
            record(f'{name}{suffix}', np.clip(pred.cpu().numpy().squeeze(), 0, 1), x0)
        del model
        torch.cuda.empty_cache()

    ens_defs = {
        'ens_x0_3sched_single': ['x0_uniform_single', 'x0_dose_unif_single', 'x0_EI_single'],
        'ens_bridge_unet_redcnn_single': ['x0_uniform_single', 'bridge_redcnn_uniform_single'],
        'ens_endpoints': ['endpoint_unet', 'endpoint_redcnn'],
    }
    for ens_name, members in ens_defs.items():
        if not all(m in singles for m in members):
            print(f"SKIP {ens_name}: missing members")
            continue
        for sid in sids:
            _, x0 = data[sid]
            avg = np.mean([singles[m][sid] for m in members], axis=0)
            record(ens_name, np.clip(avg, 0, 1), x0)
        meta[ens_name] = {'members': members}

    np.savez_compressed(
        os.path.join(out_dir, 'per_slice.npz'),
        slice_ids=np.array(sids),
        **{f'{m}__{k}': np.array(v[k]) for m, v in per.items() for k in v if v[k]})

    summary = {}
    for m, v in sorted(per.items()):
        summary[m] = {
            'psnr_mean': float(np.mean(v['psnr'])), 'psnr_std': float(np.std(v['psnr'])),
            'ssim_mean': float(np.mean(v['ssim'])), 'ssim_std': float(np.std(v['ssim'])),
            'n': len(v['psnr']),
        }
        if v['lpips']:
            summary[m]['lpips_mean'] = float(np.mean(v['lpips']))
            summary[m]['lpips_std'] = float(np.std(v['lpips']))
    result = {'dataset': args.dataset, 'alpha': alpha, 'ref_precomputed': ref,
              'n_test': len(sids), 'meta': meta, 'summary': summary}

    from scipy.stats import wilcoxon
    finals = [m for m in per if m == 'low_dose' or m.endswith('_single')
              or m.endswith('_step5') or m.startswith('endpoint')
              or m.startswith('ens_')]
    pvals = {}
    for i, m1 in enumerate(sorted(finals)):
        for m2 in sorted(finals)[i + 1:]:
            a, b = np.array(per[m1]['psnr']), np.array(per[m2]['psnr'])
            if len(a) == len(b):
                try:
                    pvals[f'{m1}|{m2}'] = {
                        'psnr_p': float(wilcoxon(a, b).pvalue),
                        'ssim_p': float(wilcoxon(np.array(per[m1]['ssim']),
                                                 np.array(per[m2]['ssim'])).pvalue),
                        'psnr_diff': float(np.mean(a) - np.mean(b)),
                    }
                except Exception:
                    pass
    result['wilcoxon'] = pvals

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*100}")
    print(f"UNIFIED EVAL v2 — {args.dataset} ({len(sids)} slices, EI-precomputed inputs)")
    hdr = f"{'method':<28s}{'PSNR':>16s}{'SSIM':>10s}"
    if lp is not None:
        hdr += f"{'LPIPS':>10s}"
    print(hdr)
    for m in sorted(summary):
        s = summary[m]
        line = (f"{m:<28s}{s['psnr_mean']:>10.2f}±{s['psnr_std']:<5.2f}"
                f"{s['ssim_mean']:>10.4f}")
        if 'lpips_mean' in s:
            line += f"{s['lpips_mean']:>10.4f}"
        print(line)
    print(f"saved -> {out_dir}")


if __name__ == '__main__':
    main()
