"""Regret of each method against the best per-dose network."""
import os, json, argparse
import numpy as np
from scipy.stats import wilcoxon

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT_ROOT = f"{DATA_ROOT}/dose_bridge_baselines"
ALPHA = {'2detect': 0.01, 'ldct': 0.1}
ONE_MODEL = {
    '2detect': [('unet_res_bridge_uniform@t', 'U-Net-res bridge (t known)'),
                ('unet_res_bridge_uniform_aug_ema@t', 'OURS U-Net bridge aug+EMA 160'),
                ('unet_res_bridge_uniform_aug_ema_lr0.0002@t', 'OURS U-Net bridge aug lr2e-4'),
                ('unet_res96_bridge_uniform_aug_ema@t', 'OURS U-Net96 bridge aug'),
                ('rep4_unet_res_bridge_uniform_aug_ema@t', 'OURS rep4 U-Net bridge aug'),
                ('rep4_unet_res_bridge_uniform_aug_ema_lr0.0002@t_alt', 'OURS rep4 U-Net bridge aug lr2e-4'),
                ('rep4_unet_res96_bridge_uniform_aug_ema_lr0.0002@t', 'OURS rep4 U-Net96 bridge aug lr2e-4'),
                ('rep4_unet_res96_ref_bridge_uniform_aug_ema_lr0.0002v2@t', 'OURS rep4v2 U-Net96+ref bridge aug lr2e-4'),
                ('rep4_unet_res96_bridge_uniform_seed123_aug_ema_lr0.0002v2@t', 'OURS rep4v2 U-Net96 seed123'),
                ('rep4_unet_res96_bridge_uniform_seed2024_aug_ema_lr0.0002v2@t', 'OURS rep4v2 U-Net96 seed2024'),
                ('rep4_unet_res96_bridge_uniform_aug_ema_lr0.0002v2@t', 'OURS rep4v2 U-Net96 bridge aug lr2e-4'),
                ('rep4_unet_res_bridge_uniform_aug_ema_lr0.0002v2@t', 'OURS rep4v2 U-Net bridge aug lr2e-4'),
                ('rep4_hybrid96_bridge_uniform_aug_ema_lr0.0002@t', 'OURS-v2 rep4 hybrid96 bridge (round 9)'),
                ('rep4_wide_film_bridge_uniform_aug_ema@t_alt', 'rep4 wide_film bridge (round 9 ablation)'),
                ('unet_res96_bridge_uniform_aug_ema_lr0.0002@t_alt', 'OURS U-Net96 bridge aug lr2e-4 ep320'),
                ('unet_res96_bridge_uniform_aug_ema_lr0.0002@t', 'OURS U-Net96 bridge aug lr2e-4'),
                ('unet_res_bridge_uniform_aug_ema_bs32@t', 'OURS U-Net bridge aug bs32'),
                ('redcnn_bridge_uniform_alt', 'RED-CNN bridge uniform ep300 (blind)'),
                ('redcnn_bridge_uniform', 'RED-CNN bridge uniform ep200 (blind)'),
                ('redcnn_bridge_equal_improvement', 'RED-CNN bridge EI ep200 (blind)'),
                ('redcnn_bridge_equal_improvement_alt', 'RED-CNN bridge EI ep300 (blind)'),
                ('redcnn_wide_bridge_uniform', 'RED-CNN-wide bridge ep200 (blind)'),
                ('redcnn_wide_bridge_uniform_alt', 'RED-CNN-wide bridge ep300 (blind)'),
                ('redcnn_wide_bridge_equal_improvement', 'RED-CNN-wide bridge EI ep200 (blind)'),
                ('redcnn_wide_bridge_equal_improvement_alt', 'RED-CNN-wide bridge EI ep300 (blind)'),
                ('redcnn_t_bridge_uniform@t', 'RED-CNN-t bridge (t known)'),
                ('drunet_bridge_uniform@t', 'DRUNet bridge (t known)'),
                ('nafnet_bridge_uniform@t', 'NAFNet bridge (t known)'),
                ('redcnn_bridge_uniform', 'RED-CNN bridge uniform (blind)'),
                ('redcnn_wide_bridge_uniform', 'RED-CNN-wide bridge (blind)'),
                ('dncnn_bridge_uniform', 'DnCNN bridge (blind)'),
                ('unet_res_interp_uniform@t', 'U-Net-res interp control (t known)'),
                ('bridge_redcnn_EI', 'RED-CNN bridge EI (blind)'),
                ('x0_uniform@t', 'paper U-Net bridge x0-uniform (t known)'),
                ('bm3d', 'BM3D (dose-oracle sigma)'),
                ('ddim_denoise', 'Diffusion prior, DDIM denoise (dose-oracle tau)')],
    'ldct': [('unet_res_bridge_uniform@t', 'U-Net-res bridge (t known)'),
             ('redcnn_bridge_equal_improvement', 'RED-CNN bridge EI ep200 (blind)'),
             ('redcnn_wide_bridge_equal_improvement', 'RED-CNN-wide bridge EI ep200 (blind)'),
             ('redcnn_t_bridge_equal_improvement@t', 'RED-CNN-t bridge EI (t known)'),
             ('unet_res_interp_uniform@t', 'U-Net-res interp control (t known)'),
             ('unet_res_bridge_uniform_aug_ema@t', 'OURS LDCT U-Net bridge aug'),
             ('unet_res_bridge_uniform_aug_ema_lr0.0002@t', 'OURS LDCT U-Net bridge aug lr2e-4'),
             ('unet_res96_bridge_uniform_aug_ema_lr0.0002@t', 'OURS LDCT U-Net96 bridge aug lr2e-4'),
             ('rep4_hybrid_bridge_uniform_aug_ema_lr0.0002@t', 'OURS-v2 LDCT rep4 hybrid bridge (round 9)'),
             ('hybrid_bridge_uniform_aug_ema_lr0.0002@t', 'LDCT hybrid bridge rep1 (round 9)'),
             ('rep4_wide_film_bridge_uniform_aug_ema@t', 'LDCT rep4 wide_film bridge (round 9 ablation)'),
             ('rep4_unet_res_bridge_uniform_aug_ema_lr0.0002@t', 'LDCT rep4 U-Net bridge (round 9)'),
             ('bm3d', 'BM3D (dose-oracle sigma)'),
             ('bridge_redcnn_uniform', 'RED-CNN bridge uniform (blind)'),
             ('bridge_redcnn_EI', 'RED-CNN bridge EI (blind)'),
             ('x0_dose_unif@t', 'paper U-Net bridge dose-unif (t known)')],
}
ENDPOINTS = {
    '2detect': [('redcnn_endpoint_uniform', 'RED-CNN endpoint'), ('redcnn_endpoint_uniform_alt', 'RED-CNN endpoint ep1000'),
                ('redcnn_endpoint_uniform', 'RED-CNN endpoint (default recipe, 100 ep; round 11)'),
                ('unet_res_endpoint_uniform@t1', 'U-Net-res endpoint (100 ep; round 11)'),
                ('drunet_endpoint_uniform@t1', 'DRUNet endpoint (100 ep; round 11)'),
                ('nafnet_endpoint_uniform@t1', 'NAFNet endpoint (100 ep; round 11)'),
                ('dncnn_endpoint_uniform', 'DnCNN endpoint (100 ep; round 11)'),
                ('edcnn_endpoint_uniform', 'EDCNN endpoint (100 ep; round 11)'),
                ('mapnn_endpoint_uniform', 'MAP-NN endpoint (100 ep; round 11)'),
                ('restormer_endpoint_uniform', 'Restormer endpoint (100 ep; round 11)'),
                ('redcnn_endpoint_uniform_aug_ema', 'RED-CNN endpoint aug ep1000'),
                ('redcnn_wide_endpoint_uniform_aug_ema', 'RED-CNN-wide endpoint aug ep1000'),
                ('edcnn_endpoint_uniform', 'EDCNN endpoint (default recipe)'),
                ('rep4_hybrid96_endpoint_uniform_aug_ema_lr0.0002@t1', 'hybrid96 endpoint rep4 (round 9 control)'),
                ('unet_res_endpoint_uniform_aug_ema@t1_alt', 'U-Net-res endpoint aug ep800'),
                ('drunet_endpoint_uniform_aug_ema@t1', 'DRUNet endpoint aug ep1000'),
                ('dncnn_endpoint_uniform_aug_ema', 'DnCNN endpoint aug ep1000'),
                ('nafnet_endpoint_uniform_aug_ema_lr0.001@t1', 'NAFNet endpoint aug ep1000'),
                ('edcnn_endpoint_uniform_aug_ema', 'EDCNN endpoint aug ep1000'),
                ('redcnn_wide_endpoint_uniform', 'RED-CNN-wide endpoint ep1000'), ('unet_res_endpoint_uniform@t1', 'U-Net-res endpoint'),
                ('drunet_endpoint_uniform@t1', 'DRUNet endpoint'), ('nafnet_endpoint_uniform@t1', 'NAFNet endpoint'),
                ('redcnn_wide_endpoint_uniform', 'RED-CNN-wide endpoint'), ('dncnn_endpoint_uniform', 'DnCNN endpoint'),
                ('endpoint_unet', 'paper U-Net endpoint')],
    'ldct': [('endpoint_redcnn', 'RED-CNN endpoint'), ('redcnn_endpoint_equal_improvement_alt', 'RED-CNN endpoint ep1000'),
             ('redcnn_endpoint_equal_improvement', 'RED-CNN endpoint (default recipe, 100 ep; round 11)'),
             ('unet_res_endpoint_equal_improvement@t1', 'U-Net-res endpoint (100 ep; round 11)'),
             ('drunet_endpoint_equal_improvement@t1', 'DRUNet endpoint (100 ep; round 11)'),
             ('nafnet_endpoint_equal_improvement@t1', 'NAFNet endpoint (100 ep; round 11)'),
             ('dncnn_endpoint_equal_improvement', 'DnCNN endpoint (100 ep; round 11)'),
             ('edcnn_endpoint_equal_improvement', 'EDCNN endpoint (100 ep; round 11)'),
             ('mapnn_endpoint_equal_improvement', 'MAP-NN endpoint (100 ep; round 11)'),
             ('restormer_endpoint_equal_improvement', 'Restormer endpoint (100 ep; round 11)'),
             ('redcnn_endpoint_equal_improvement_aug_ema', 'RED-CNN endpoint aug ep1000'),
             ('unet_res_endpoint_uniform_aug_ema@t1', 'U-Net-res endpoint aug ep1000'),
             ('edcnn_endpoint_equal_improvement_aug_ema', 'EDCNN endpoint aug ep1000'),
             ('redcnn_wide_endpoint_equal_improvement_aug_ema', 'RED-CNN-wide endpoint aug ep1000'),
             ('redcnn_endpoint_equal_improvement', 'RED-CNN endpoint (default recipe, 400 ep)'),
             ('edcnn_endpoint_equal_improvement', 'EDCNN endpoint (default recipe)'),
             ('dncnn_endpoint_equal_improvement', 'DnCNN endpoint (default recipe)'),
             ('nafnet_endpoint_equal_improvement@t1', 'NAFNet endpoint (default recipe)'),
             ('drunet_endpoint_equal_improvement@t1', 'DRUNet endpoint (default recipe)'),
             ('unet_res_endpoint_uniform@t1', 'U-Net-res endpoint (default recipe)'),
             ('hybrid_endpoint_uniform_aug_ema_lr0.0002@t1', 'hybrid endpoint (round 9 control)'),
             ('redcnn_wide_endpoint_equal_improvement', 'RED-CNN-wide endpoint ep1000'), ('unet_res_endpoint_uniform@t1', 'U-Net-res endpoint'),
             ('endpoint_unet', 'paper U-Net endpoint')],
}


def _pub_ok(m):
    """BASE_PUB=1: only default-recipe baselines may serve as specialists / single-dose references"""
    if os.environ.get('BASE_PUB') != '1':
        return True
    return not any(x in m for x in ('aug', 'wide', 'hybrid', 'rep4', 'rep5', 'ref'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['2detect', 'ldct'])
    ap.add_argument('--sweep', default=None)
    args = ap.parse_args()
    d = args.sweep or f'{OUT_ROOT}/eval_sweep_{args.dataset}_v1'
    S = json.load(open(f'{d}/summary.json'))
    P = dict(np.load(f'{d}/per_slice.npz'))
    from glob import glob
    for ef in sorted(glob(f'{d}/extra_*.json')):
        E = json.load(open(ef))
        for m, byk in E['models'].items():
            S['models'].setdefault(m, {}).update(byk)
            for k in byk:
                npy = f'{d}/{m}_{k}_psnr.npy'
                if os.path.exists(npy):
                    P[f'{m}|{k}|psnr'] = np.load(npy)
        S['knot_t'].update({k: v for k, v in E.get('knot_t', {}).items() if k not in S['knot_t']})
        print(f"merged {os.path.basename(ef)}: {list(E['models'])}")
    a = ALPHA[args.dataset]
    knots = sorted(S['knot_t'].items(), key=lambda kv: kv[1])
    seen, uk = set(), []
    for k, t in knots:
        if round(t, 3) in seen:
            continue
        seen.add(round(t, 3)); uk.append((k, t))
    models = S['models']

    def arr(m, k):
        key = f'{m}|{k}|psnr'
        return P[key] if key in P else None

    FAMILY = {'uniform': 'uniform', 'EI': 'equal_improvement'}

    def specialists_at(k):
        fam, step = k.split('_step')
        if fam not in FAMILY:
            return []
        sched = FAMILY[fam]
        out = []
        for m in models:
            if (f'_spec_{sched}_k{step}' in m or f'_endpoint_{sched}_k{step}' in m) and k in models[m] and '@t1' not in m and _pub_ok(m):
                out.append(m)
        if step == '4':
            for m, _ in ENDPOINTS[args.dataset]:
                if k in models.get(m, {}) and 'paper' not in _ and _pub_ok(m):
                    out.append(m)
        return out

    rows = []
    print(f"\n### {args.dataset}: regret vs best per-dose specialist (dB; negative = below specialist)")
    hdr = f"{'knot':14s}{'dose%':>7s}{'input':>7s}{'best spec':>26s}"
    print(hdr)
    stats = {}
    for k, t in uk:
        dose = 100 * a ** t
        specs = specialists_at(k)
        if not specs:
            continue
        best = max(specs, key=lambda m: models[m][k]['psnr'])
        bs = arr(best, k)
        line = f"{k:14s}{dose:7.1f}{models['input'][k]['psnr']:7.2f}{best[:24]:>26s}={models[best][k]['psnr']:.2f}"
        stats[k] = {'t': t, 'dose_pct': dose, 'best_specialist': best, 'best_specialist_psnr': models[best][k]['psnr']}
        for m, lab in ONE_MODEL[args.dataset] + ENDPOINTS[args.dataset]:
            x = arr(m, k)
            if x is None or bs is None:
                continue
            diff = x - bs
            p = wilcoxon(x, bs).pvalue if np.any(diff != 0) else 1.0
            stats[k][m] = {'psnr': float(x.mean()), 'regret': float(diff.mean()), 'p_vs_spec': float(p),
                           'win_rate_vs_spec': float(np.mean(diff > 0)),
                           'gain_over_input': float(x.mean() - models['input'][k]['psnr'])}
            line += f"\n    {lab:36s} {x.mean():6.2f}  regret {diff.mean():+6.2f}  p={p:.1e}  win {np.mean(diff>0)*100:4.0f}%  Δin {x.mean()-models['input'][k]['psnr']:+6.2f}"
        print(line)
    og = [(k, t) for k, t in uk if k.startswith('offgrid_step')]
    if og:
        print(f"\n### {args.dataset}: off-grid test doses (PSNR dB)")
        print(f"{'model':44s}" + "".join(f"{100*a**t:>8.1f}%" for _, t in og))
        print(f"{'input':44s}" + "".join(f"{models['input'][k]['psnr']:9.2f}" for k, _ in og))
        for m, lab in ONE_MODEL[args.dataset] + ENDPOINTS[args.dataset]:
            if m in models and all(k in models[m] for k, _ in og):
                print(f"{lab[:44]:44s}" + "".join(f"{models[m][k]['psnr']:9.2f}" for k, _ in og))
        specs = sorted(m for m in models if '_spec_uniform_k' in m and '@t1' not in m)
        for m in specs:
            if all(k in models[m] for k, _ in og):
                print(f"{m[:44]:44s}" + "".join(f"{models[m][k]['psnr']:9.2f}" for k, _ in og))
    # summary lines for the abstract
    print("\n### headline numbers")
    for m, lab in ONE_MODEL[args.dataset]:
        regs = [stats[k][m]['regret'] for k in stats if m in stats[k]]
        if regs:
            print(f"{lab:36s}: worst regret {min(regs):+.2f} dB, mean {np.mean(regs):+.2f}, over {len(regs)} doses")
    for m, lab in ENDPOINTS[args.dataset]:
        regs = [stats[k][m]['regret'] for k in stats if m in stats[k]]
        gins = [stats[k][m]['gain_over_input'] for k in stats if m in stats[k]]
        if regs:
            print(f"{lab:36s}: worst regret {min(regs):+.2f} dB; below input at {sum(g < 0 for g in gins)}/{len(gins)} doses (worst Δin {min(gins):+.2f})")
    with open(f'{d}/regret_stats.json', 'w') as f:
        json.dump(stats, f, indent=1)


if __name__ == '__main__':
    main()
