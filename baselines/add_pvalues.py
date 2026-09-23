"""Paired Wilcoxon tests between methods from the per-slice metrics."""
import json
import numpy as np
from scipy.stats import wilcoxon

DATA_ROOT = os.environ.get("DOSE_BRIDGE_DATA", "data")
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROOT = f"{DATA_ROOT}/dose_bridge_baselines"

for name in ['eval_2detect', 'eval_ldct', 'eval_real_ldct', 'eval_real_2detect']:
    d = np.load(f'{ROOT}/{name}/per_slice.npz')
    methods = sorted({k.rsplit('__', 1)[0] for k in d.files if '__' in k})
    summ_path = f'{ROOT}/{name}/summary.json'
    summ = json.load(open(summ_path))
    pv = {}
    for i, m1 in enumerate(methods):
        for m2 in methods[i + 1:]:
            entry = {}
            for metric in ['psnr', 'ssim', 'lpips']:
                k1, k2 = f'{m1}__{metric}', f'{m2}__{metric}'
                if k1 in d.files and k2 in d.files and len(d[k1]) == len(d[k2]):
                    try:
                        entry[f'{metric}_p'] = float(wilcoxon(d[k1], d[k2]).pvalue)
                        entry[f'{metric}_diff'] = float(np.mean(d[k1]) - np.mean(d[k2]))
                    except Exception:
                        pass
            if entry:
                pv[f'{m1}|{m2}'] = entry
    summ['wilcoxon'] = pv
    json.dump(summ, open(summ_path, 'w'), indent=2)
    print(name, 'pairs:', len(pv))
print('done')
