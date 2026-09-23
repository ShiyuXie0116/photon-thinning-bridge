"""Builds the result tables from the evaluation summaries."""
import os, json, argparse
import numpy as np

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

R = f"{DATA_ROOT}/dose_bridge_baselines"
PAPER = os.environ.get('PAPER_DIR', f"{CODE_ROOT}/ICASSP2027/v7")


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def merged_models(sweep_dir, v1_dir):
    S = load(f'{sweep_dir}/summary.json'); M = dict(S['models']) if S else {}
    for d in (sweep_dir, v1_dir):
        for name in ('extra_bm3d_EI.json', 'extra_bm3d.json', 'extra_diffusion_EI.json', 'extra_diffusion.json',
                     'extra_coredif_aug.json', 'extra_coredif.json'):
            E = load(f'{d}/{name}')
            if E:
                for m, byk in E['models'].items():
                    M.setdefault(m, {}).update({k: v for k, v in byk.items() if k not in M.get(m, {})})
    return M


def pick(M, keys, knot='EI_step4'):
    """first existing key (priority order) -> (key, metrics) or (None, None); DDS lives on the B realization"""
    for k in keys:
        if k in M and knot in M[k]:
            return k, M[k][knot]
        if k in M and k == 'dds' and 'B_step4' in M[k]:
            return k, M[k]['B_step4']
    return None, None


def cell3(v, bold=()):
    if v is None:
        return '-- & -- & --'
    out = []
    for m, fmt in (('psnr', '{:.2f}'), ('ssim', '{:.3f}'), ('lpips', '{:.3f}')):
        x = v.get(m)
        s = fmt.format(x) if x is not None else '--'
        out.append(r'\textbf{' + s + '}' if m in bold and x is not None else s)
    return ' & '.join(out)


ROWS = [
    ('Low-dose input', ['input'], ['input']),
    ('BM3D (oracle $\\sigma$) \\cite{dabov2007bm3d}', ['bm3d'], ['bm3d']),
    ('Diffusion prior, DDIM \\cite{song2021ddim}', ['ddim_denoise'], ['ddim_denoise']),
    ('CoreDiff \\cite{gao2024corediff}', ['coredif_aug', 'coredif'], ['coredif_aug', 'coredif']),
    ('DnCNN \\cite{zhang2017dncnn}', ['dncnn_endpoint_uniform_aug_ema_ep1000', 'dncnn_endpoint_uniform'], ['dncnn_endpoint_equal_improvement_aug_ema_ep1000']),
    ('EDCNN \\cite{liang2020edcnn}', ['edcnn_endpoint_uniform_aug_ema_ep1000'], ['edcnn_endpoint_equal_improvement_aug_ema_ep1000']),
    ('NAFNet \\cite{chen2022nafnet}', ['nafnet_endpoint_uniform_aug_ema_lr0.001_ep1000@t1', 'nafnet_endpoint_uniform@t1'], ['nafnet_endpoint_equal_improvement_aug_ema_lr0.001_ep1000@t1']),
    ('DRUNet \\cite{zhang2021drunet}', ['drunet_endpoint_uniform_aug_ema_ep1000@t1', 'drunet_endpoint_uniform@t1'], ['drunet_endpoint_equal_improvement_aug_ema_ep1000@t1']),
    ('U-Net, single dose \\cite{jin2017deep}', ['rep4_unet_res_endpoint_uniform_aug_ema_lr0.0002_v2@t1', 'unet_res_endpoint_uniform_aug_ema_ep800@t1', 'unet_res_endpoint_uniform_aug_ema@t1', 'unet_res_endpoint_uniform@t1'],
     ['unet_res_endpoint_uniform_aug_ema_ep1000@t1', 'unet_res_endpoint_uniform@t1']),
    ('RED-CNN \\cite{chen2017low}', ['redcnn_endpoint_uniform_aug_ema_ep1000', 'redcnn_endpoint_uniform_ep1000', 'redcnn_endpoint_uniform'],
     ['redcnn_endpoint_equal_improvement_aug_ema_ep1000', 'redcnn_endpoint_equal_improvement_ep1000', 'endpoint_redcnn']),
    ('RED-CNN-wide', ['redcnn_wide_endpoint_uniform_aug_ema_ep1000', 'redcnn_wide_endpoint_uniform_ep1000'],
     ['redcnn_wide_endpoint_equal_improvement_aug_ema_ep1000', 'redcnn_wide_endpoint_equal_improvement_ep1000']),
]
BASE_PUB = os.environ.get('BASE_PUB') == '1'
if BASE_PUB:
    ROWS = [
        ('Low-dose input', ['input'], ['input']),
        ('BM3D (oracle $\\sigma$) \\cite{dabov2007bm3d}', ['bm3d'], ['bm3d']),
        ('Diffusion prior, DDIM \\cite{song2021ddim}', ['ddim_denoise'], ['ddim_denoise']),
        ('CoreDiff \\cite{gao2024corediff}', ['coredif_aug', 'coredif'], ['coredif_aug', 'coredif']),
        ('DnCNN \\cite{zhang2017dncnn}', ['dncnn_endpoint_uniform'], ['dncnn_endpoint_equal_improvement']),
        ('EDCNN \\cite{liang2020edcnn}', ['edcnn_endpoint_uniform'], ['edcnn_endpoint_equal_improvement']),
        ('NAFNet \\cite{chen2022nafnet}', ['nafnet_endpoint_uniform@t1'], ['nafnet_endpoint_equal_improvement@t1']),
        ('DRUNet \\cite{zhang2021drunet}', ['drunet_endpoint_uniform@t1'], ['drunet_endpoint_equal_improvement@t1']),
        ('U-Net, single dose \\cite{jin2017deep}', ['unet_res_endpoint_uniform@t1'], ['unet_res_endpoint_uniform@t1']),
        ('RED-CNN \\cite{chen2017low}', ['redcnn_endpoint_uniform'], ['redcnn_endpoint_equal_improvement', 'redcnn_endpoint_equal_improvement_ep1000']),
        ('WGAN-VGG \\cite{yang2018low}', ['wganvgg_endpoint_uniform_pub', 'wganvgg_endpoint_uniform_it30k', 'wganvgg_endpoint_uniform'], ['wganvgg_endpoint_equal_improvement_pub', 'wganvgg_endpoint_equal_improvement']),
        ('MAP-NN \\cite{shan2019competitive}', ['mapnn_endpoint_uniform'], ['mapnn_endpoint_equal_improvement']),
        ('Restormer \\cite{zamir2022restormer}', ['restormer_endpoint_uniform'], ['restormer_endpoint_equal_improvement_ep400', 'restormer_endpoint_equal_improvement']),
    ]
BASE_TAG = os.environ.get('BASE_TAG', '')
def _tagged(keys, ldct):
    if not BASE_TAG:
        return keys
    out = []
    for k in keys:
        base, _, suf = k.partition('@')
        if not (base.startswith('') and '_endpoint_' in base) or 'wganvgg' in base:
            out.append(k); continue
        if base.endswith('_ep1000'):
            base = base[:-7]
        if ldct:
            base = base.replace('_endpoint_uniform', '_endpoint_equal_improvement')
        t = f'{base}_{BASE_TAG}' + (f'@{suf}' if suf else '')
        if t not in out:
            out.append(t)
    return out
if BASE_PUB and BASE_TAG:
    ROWS = [(lab, _tagged(a, False), _tagged(b, True)) for lab, a, b in ROWS]
OURS_2D = ([os.environ['OURS2D_KEY']] if os.environ.get('OURS2D_KEY') else []) + \
          ['rep4_unet_res96_bridge_uniform_aug_ema_lr0.0002_ep80v2@t',
           'unet_res96_bridge_uniform_aug_ema_lr0.0002_ep320@t', 'rep4_unet_res_bridge_uniform_aug_ema_lr0.0002_ep80v2@t',
           'rep4_unet_res96_bridge_uniform_aug_ema_lr0.0002_ep80@t', 'rep4_unet_res_bridge_uniform_aug_ema_lr0.0002_ep80@t',
           'unet_res96_bridge_uniform_aug_ema_lr0.0002_ep160@t', 'unet_res_bridge_uniform_aug_ema_lr0.0002_ep160@t',
           'unet_res_bridge_uniform_aug_ema_bs32_ep160@t', 'unet_res_bridge_uniform_aug_ema_ep160@t', 'unet_res_bridge_uniform@t']
OURS_LD = ([os.environ['OURSLD_KEY']] if os.environ.get('OURSLD_KEY') else []) + \
          ['unet_res_bridge_uniform_aug_ema_lr0.0002_ep160@t',
           'unet_res_bridge_uniform_aug_ema_ep160@t', 'unet_res_bridge_uniform@t']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep2d', default=f'{R}/eval_sweep_2detect_v3')
    ap.add_argument('--sweepld', default=f'{R}/eval_sweep_ldct_v4' if os.path.exists(f'{R}/eval_sweep_ldct_v4/summary.json') else f'{R}/eval_sweep_ldct_v3')
    ap.add_argument('--ours2d', default=None, help='override: sweep key for Ours on 2DeteCT')
    ap.add_argument('--oursld', default=None)
    ap.add_argument('--ms2d', default='ures96_rep4_v2', help='eval_bridge_samplers2 tag for the 2DeteCT multi-step row (must be the single-step checkpoint)')
    ap.add_argument('--msld', default='ldct_ures_lr2e4', help='eval_bridge_samplers2 tag for the LDCT multi-step row')
    args = ap.parse_args()
    for d in (args.sweep2d, args.sweepld):
        if not os.path.exists(f'{d}/summary.json'):
            print('missing', d, '-> falling back to v2');
    s2d = args.sweep2d if os.path.exists(f'{args.sweep2d}/summary.json') else f'{R}/eval_sweep_2detect_v2'
    sld = args.sweepld if os.path.exists(f'{args.sweepld}/summary.json') else f'{R}/eval_sweep_ldct_v2'
    M2 = merged_models(s2d, f'{R}/eval_sweep_2detect_v1'); ML = merged_models(sld, f'{R}/eval_sweep_ldct_v1')
    def best(M, cands, override):
        if override:
            return override, M[override]['EI_step4']
        for k in cands:
            if k in M and 'EI_step4' in M[k]:
                return k, M[k]['EI_step4']
        return None, None
    o2k, o2 = best(M2, OURS_2D, args.ours2d); olk, ol = best(ML, OURS_LD, args.oursld)
    print('Ours 2DeteCT =', o2k, o2); print('Ours LDCT   =', olk, ol)
    def multistep(tag, key='pmavg_EI5'):
        S = load(f'{R}/eval_bridge_samplers2/{tag}/summary.json')
        if S and key in S['summary']:
            v = S['summary'][key]; return S['ckpt'], {'psnr': v['psnr'], 'ssim': v['ssim'], 'lpips': v['lpips'], 'n': v['n']}
        return None, None
    m2k, m2 = None, None
    for alt in (args.ms2d, 'ures96_rep4_v2', 'ures96_lr2e4', 'ures_lr2e4'):   # same checkpoint as the single-step row (--ms2d first)
        k_, v_ = multistep(alt)
        if v_:
            m2k, m2 = k_, v_; break
    mlk, ml = multistep(args.msld)
    print('Ours 2DeteCT multi-step =', m2k, m2); print('Ours LDCT multi-step   =', mlk, ml)
    rows = []
    for lab, k2, kl in ROWS:
        a, v2 = pick(M2, k2); b, vl = pick(ML, kl)
        rows.append((lab, v2, vl, a, b))
    rows.append(('\\textbf{Ours}' if os.environ.get('OURS_5STEP_ONLY') == '1' else '\\textbf{Ours}: 5-step posterior-mean bridge', m2 or o2, ml or ol, m2k or o2k, mlk or olk))
    if os.environ.get('OURS_5STEP_ONLY') != '1':
        rows.append(('\\textbf{Ours}, single step', o2, ol, o2k, olk))
    # bold the best PSNR/SSIM/LPIPS per column (excluding input)
    def winners(idx, metric, larger):
        nd = 2 if metric == 'psnr' else 3
        vals = [(i, round(r[idx][metric], nd)) for i, r in enumerate(rows) if i > 0 and r[idx] and r[idx].get(metric) is not None]
        if not vals: return set()
        best = max(vals, key=lambda t: t[1])[1] if larger else min(vals, key=lambda t: t[1])[1]
        return {i for i, v in vals if v == best}
    W = {(idx, m): winners(idx, m, m != 'lpips') for idx in (1, 2) for m in ('psnr', 'ssim', 'lpips')}
    lines = []
    for i, (lab, v2, vl, a, b) in enumerate(rows):
        b2 = tuple(m for m in ('psnr', 'ssim', 'lpips') if i in W[(1, m)]); bl = tuple(m for m in ('psnr', 'ssim', 'lpips') if i in W[(2, m)])
        lines.append(f"{lab} & {cell3(v2, b2)} & {cell3(vl, bl)} \\\\")
    lines.insert(-(1 if os.environ.get('OURS_5STEP_ONLY') == '1' else 2), r'\midrule')
    tab1 = "\n".join(lines)
    with open(f'{PAPER}/tab1_methods.tex', 'w') as f:
        f.write('% generated by make_tables.py\n' + tab1 + '\n')
    print('\n' + tab1)
    with open(f'{PAPER}/tab1_mapping.txt', 'w') as f:
        for lab, v2, vl, a, b in rows:
            f.write(f"{lab}\t2D={a}\tLD={b}\n")
    def regret_rows(sweep, keys_by_label):
        G = load(f'{sweep}/regret_stats.json')
        if not G: return {}
        ks = sorted([k for k in G if '_step' in k and not k.startswith(('offgrid', 'B_'))], key=lambda k: G[k]['t'])
        out = {}
        for lab, cands in keys_by_label:
            for m in cands:
                r = [G[k][m]['regret'] for k in ks if m in G[k]]
                g = [G[k][m]['gain_over_input'] for k in ks if m in G[k]]
                if r:
                    out[lab] = (min(r), float(np.mean(r)), sum(x < 0 for x in g), len(g)); break
        return out
    T2 = [('\\textbf{Ours}' if os.environ.get('OURS_5STEP_ONLY') == '1' else '\\textbf{Ours} (one checkpoint)', ([o2k] if o2k else []) + OURS_2D, ([olk] if olk else []) + OURS_LD),
          ('RED-CNN, single dose', ['redcnn_endpoint_uniform_aug_ema_ep1000', 'redcnn_endpoint_uniform_ep1000'], ['redcnn_endpoint_equal_improvement_aug_ema_ep1000', 'redcnn_endpoint_equal_improvement_ep1000', 'endpoint_redcnn']),
          ('DRUNet, single dose', ['drunet_endpoint_uniform_aug_ema_ep1000@t1', 'drunet_endpoint_uniform@t1'], ['drunet_endpoint_equal_improvement_aug_ema_ep1000@t1']),
          ('U-Net, single dose', ['unet_res_endpoint_uniform_aug_ema_ep800@t1', 'unet_res_endpoint_uniform@t1'], ['unet_res_endpoint_uniform_aug_ema_ep1000@t1', 'unet_res_endpoint_uniform@t1']),
          ('BM3D (oracle $\\sigma$)', ['bm3d'], ['bm3d'])]
    if BASE_PUB:
        T2 = [('\\textbf{Ours}' if os.environ.get('OURS_5STEP_ONLY') == '1' else '\\textbf{Ours} (one checkpoint)', ([o2k] if o2k else []) + OURS_2D, ([olk] if olk else []) + OURS_LD),
              ('RED-CNN, single dose', ['redcnn_endpoint_uniform'], ['redcnn_endpoint_equal_improvement', 'redcnn_endpoint_equal_improvement_ep1000']),
              ('DRUNet, single dose', ['drunet_endpoint_uniform@t1'], ['drunet_endpoint_equal_improvement@t1']),
              ('U-Net, single dose', ['unet_res_endpoint_uniform@t1'], ['unet_res_endpoint_uniform@t1']),
              ('BM3D (oracle $\\sigma$)', ['bm3d'], ['bm3d'])]
        T2 = [(l, _tagged(a, False), _tagged(b, True)) for l, a, b in T2]
    r2 = regret_rows(s2d, [(l, a) for l, a, _ in T2]); rl = regret_rows(sld, [(l, b) for l, _, b in T2])
    l2 = []
    for lab, _, _ in T2:
        a = r2.get(lab); b = rl.get(lab)
        ca = f"${a[0]:+.2f}$ & ${a[1]:+.2f}$ & {a[2]}/{a[3]}" if a else "-- & -- & --"
        cb = f"${b[0]:+.2f}$ & ${b[1]:+.2f}$" if b else "-- & --"
        l2.append(f"{lab} & {ca} & {cb} \\\\")
    tab2 = "\n".join(l2)
    with open(f'{PAPER}/tab2_regret.tex', 'w') as f:
        f.write('% generated by make_tables.py\n' + tab2 + '\n')
    print('\n' + tab2)
    SL = load(f'{R}/eval_real_ldct/summary.json'); SL = SL.get('summary', SL) if SL else {}
    S2 = load(f'{R}/eval_real_2detect/summary.json'); S2 = S2.get('summary', S2) if S2 else {}
    SM = load(f'{R}/eval_sampler_real_2detect/summary.json'); SM = SM['summary'] if SM else {}
    def rc(v, bold=()):
        if not v: return '-- & -- & --'
        p = f"{v['psnr_mean']:.2f}{{\\scriptsize$\\pm${v['psnr_std']:.2f}}}"; s_ = f"{v['ssim_mean']:.3f}"; l = f"{v.get('lpips_mean', float('nan')):.3f}"
        return ' & '.join(r'\textbf{' + x + '}' if m in bold else x for m, x in (('psnr', p), ('ssim', s_), ('lpips', l)))
    def first(S, keys):
        for k in keys:
            if k in S: return S[k]
        return None
    T3 = [('Low-dose input', ['low_dose'], ['low_dose']),
          ('U-Net, single dose', ['endpoint_unet_res_aug_ep1000', 'endpoint_unet_res'], ['endpoint_unet_res_effI0_aug_ep1000', 'endpoint_unet_res_effI0']),
          ('RED-CNN', ['endpoint_redcnn_aug_ep1000', 'endpoint_redcnn_ep1000', 'endpoint_redcnn'], ['endpoint_redcnn_effI0_aug_ep1000', 'endpoint_redcnn_effI0_ep1000', 'endpoint_redcnn_effI0']),
          ('RED-CNN-wide', ['endpoint_redcnn_wide_aug_ep1000', 'endpoint_redcnn_wide_ep1000'], ['endpoint_redcnn_wide_effI0_aug_ep1000', 'endpoint_redcnn_wide_effI0_ep1000']),
          ('\\textbf{Ours}, 5-step posterior mean', ['__sampler_ld__'], ['__sampler__']),
          ('\\textbf{Ours}, single step', (['ours_hybrid_rep4_single'] if os.environ.get('OURS_HYBRID') == '1' else []) +
           ['ours_unet_res_aug_lr2e4_single', 'ours_unet_res_aug_single', 'bridge_unet_res_single'],
           (['ours_hybrid96_effI0_aug_lr2e4_single'] if os.environ.get('OURS_HYBRID') == '1' else []) +
           (['ours_unet96ref_effI0_aug_lr2e4'] if 'ref' in os.environ.get('OURS2D_KEY', '') else []) +
           ['ours_unet96_effI0_aug_lr2e4', 'ours_unet96_effI0_aug_lr2e4_single', 'ours_unet_res_effI0_aug_single', 'bridge_unet_res_effI0_single'])]
    if BASE_PUB:
        T3 = [('Low-dose input', ['low_dose'], ['low_dose']),
              ('U-Net, single dose', ['endpoint_unet_res'], ['endpoint_unet_res_effI0']),
              ('RED-CNN', ['endpoint_redcnn_r6ep400', 'endpoint_redcnn_ep1000', 'endpoint_redcnn'], ['endpoint_redcnn_effI0', 'endpoint_redcnn_effI0_ep1000']),
              ('\\textbf{Ours}, 5-step posterior mean', ['__sampler_ld__'], ['__sampler__']),
              ('\\textbf{Ours}, single step', (['ours_hybrid_rep4_single'] if os.environ.get('OURS_HYBRID') == '1' else []) +
               ['ours_unet_res_aug_lr2e4_single', 'ours_unet_res_aug_single', 'bridge_unet_res_single'],
               (['ours_hybrid96_effI0_aug_lr2e4_single'] if os.environ.get('OURS_HYBRID') == '1' else []) +
               ['ours_unet96_effI0_aug_lr2e4', 'ours_unet96_effI0_aug_lr2e4_single', 'ours_unet_res_effI0_aug_single', 'bridge_unet_res_effI0_single'])]
    if os.environ.get('OURS_5STEP_ONLY') == '1':
        T3 = [r for r in T3 if 'single step' not in r[0]]
        T3 = [(('\\textbf{Ours}' if r[0].startswith('\\textbf{Ours}, 5-step') else r[0]), r[1], r[2]) for r in T3]
    if BASE_PUB and BASE_TAG:
        T3[1] = ('U-Net, single dose', [f'endpoint_unet_res_{BASE_TAG}'], [f'endpoint_unet_res_effI0_{BASE_TAG}'])
        T3[2] = ('RED-CNN', [f'endpoint_redcnn_r6{BASE_TAG}'], [f'endpoint_redcnn_effI0_{BASE_TAG}'])
        if os.environ.get('T3_ALL') == '1':
            extra = [('DnCNN', 'dncnn'), ('EDCNN', 'edcnn'), ('NAFNet', 'nafnet'), ('DRUNet', 'drunet'), ('MAP-NN', 'mapnn'), ('Restormer', 'restormer')]
            rows_extra = [(lab, [f'endpoint_{a}_{BASE_TAG}'], [f'endpoint_{a}_effI0_{BASE_TAG}']) for lab, a in extra]
            rows_extra.append(('WGAN-VGG', ['endpoint_wganvgg_pub'], ['endpoint_wganvgg_effI0_pub']))
            T3 = T3[:1] + rows_extra + T3[1:]
    SML = load(f'{R}/eval_sampler_real_ldct/summary.json'); SML = SML['summary'] if SML else {}
    def that(fn, key):
        J = load(fn); v = J['summary'].get(key) if J else None
        return {'psnr_mean': v['psnr'], 'psnr_std': v['psnr_std'], 'ssim_mean': v['ssim'], 'lpips_mean': v['lpips']} if v else None
    l3 = []
    for lab, kl, k2 in T3:
        vl = first(SML, (['ldhyb4_real_var_N5_final'] if os.environ.get('OURS_HYBRID') == '1' else []) + ['ures_aug_lr2e4_var_N5_final', 'ures_aug_var_N5_final']) if kl == ['__sampler_ld__'] else first(SL, kl)
        if k2 == ['__sampler__']:
            v2 = first(SM, (['hyb96_effI0_aug_var_N5_final'] if os.environ.get('OURS_HYBRID') == '1' else []) +
                       (['ures96ref_effI0_aug_var_N5_final'] if 'ref' in os.environ.get('OURS2D_KEY', '') else []) +
                       ['ures96_effI0_aug_var_N5_final', 'ures_effI0_aug_var_N5_final', 'ures_effI0_var_N5_final'])
        else:
            v2 = first(S2, k2)
        l3.append(f"{lab} & {rc(vl)} & {rc(v2)} \\\\")
    if os.environ.get('THAT_ROW') == '1':
        vl = that(f'{R}/eval_real_ldct_tsweep/ldct_that.json', 'pmavg5_that_patient')
        v2 = that(f'{R}/eval_real_ldct_tsweep/2detect_that.json', 'pmavg5_that_patient')
        l3.append(f"\\textbf{{Ours}}, query at $\\hat t$ & {rc(vl)} & {rc(v2)} \\\\")
    tab3 = "\n".join(l3)
    with open(f'{PAPER}/tab3_real.tex', 'w') as f:
        f.write('% generated by make_tables.py\n' + tab3 + '\n')
    print('\n' + tab3)


if __name__ == '__main__':
    main()
