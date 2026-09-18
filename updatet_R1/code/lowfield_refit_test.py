"""Model-adequacy test: per-loop Jiles-Atherton fit of the 55 kA/m loops with an objective that
weights the low-field (hysteretic, |H| < 3 kA/m) window equally with the full loop, for alpha = 0
and alpha free, at -38, -10 and +15 C. Uses ja_guarded.single_loop_g (time-capped solver).
Output: results/pilot_refit/pilot_results.json. Runtime: a few minutes on a multi-core machine."""
import os, sys, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ja_model import calc_metrics, MU0
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ja_guarded import single_loop_g as single_loop, TooSlow
from scipy.optimize import differential_evolution
d = np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'gd_processed.npz'), allow_pickle=True)
T = d['temperatures']
HLOW = 3000.0

def cycle_idx(H):
    hi = np.where(H >= 0.985 * H.max())[0]; i0 = hi[0]
    gaps = np.where(np.diff(hi) > 500)[0]
    i1 = hi[gaps[0] + 1] if len(gaps) else len(H) - 1
    return i0, i1

def hc_br(H, B):
    hc = []; br = []
    for j in range(1, len(B)):
        if B[j-1]*B[j] < 0:
            t = abs(B[j-1])/(abs(B[j-1])+abs(B[j])); hc.append(abs(H[j-1]+t*(H[j]-H[j-1])))
        if H[j-1]*H[j] < 0:
            t = abs(H[j-1])/(abs(H[j-1])+abs(H[j])); br.append(abs(B[j-1]+t*(B[j]-B[j-1])))
    return (np.mean(hc) if hc else np.nan), (np.mean(br) if br else np.nan)

class Obj:
    def __init__(self, H, B, M0, free_alpha):
        self.H, self.B, self.M0, self.fa = H, B, M0, free_alpha
        self.low = np.abs(H) < HLOW
        self.sst = np.sum((B - B.mean())**2); self.sstl = np.sum((B[self.low] - B[self.low].mean())**2)
    def __call__(self, p):
        a, k, c, Ms = p[:4]; al = p[4] if self.fa else 0.0
        if a <= 0 or k <= 0 or c < 0 or c > 0.95 or Ms <= 0: return 1e30
        try: _, Bs = single_loop(a, k, c, Ms, al, self.H, M0=self.M0)
        except Exception: return 1e30
        if not np.all(np.isfinite(Bs)): return 1e30
        r = self.B - Bs
        return np.sum(r**2)/self.sst + np.sum(r[self.low]**2)/self.sstl

if __name__ == '__main__':
    out = []
    for i in [0, 6, 11]:
        H = np.asarray(d['major_H'][i], float); B = np.asarray(d['major_B'][i], float)
        M0 = B[0]/MU0 - H[0]
        i0, i1 = cycle_idx(H)
        Hc_m, Br_m = hc_br(H[i0:i1+1], B[i0:i1+1])
        Ms_tip = B.max()/MU0 - H.max()
        for fa in (False, True):
            bounds = [(500, 15000), (30, 2000), (0.0, 0.9), (Ms_tip*0.95, Ms_tip*1.3)] + ([(0.0, 3e-3)] if fa else [])
            t0 = time.time()
            res = differential_evolution(Obj(H, B, M0, fa), bounds, popsize=10, maxiter=30, tol=1e-8, seed=1,
                                         polish=True, workers=-1, updating='deferred')
            p = list(res.x) + ([] if fa else [0.0])
            _, Bs = single_loop(*p[:5], H, M0=M0)
            m = calc_metrics(B, Bs, 5 if fa else 4)
            low = np.abs(H) < HLOW
            r2low = 1 - np.sum((B[low]-Bs[low])**2)/np.sum((B[low]-B[low].mean())**2)
            Hc_s, Br_s = hc_br(H[i0:i1+1], Bs[i0:i1+1])
            rec = dict(T=float(T[i]), alpha_free=fa, a=p[0], k=p[1], c=p[2], Ms=p[3], alpha=p[4],
                       r2_adj=m['r2_adj'], rmse_mT=m['rmse']*1e3, r2_low=r2low, Hc_meas=Hc_m, Hc_model=Hc_s,
                       Br_meas=Br_m, Br_model=Br_s, seconds=time.time()-t0)
            out.append(rec)
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'pilot_refit', 'pilot_results.json'), 'w'), indent=1)
    print('DONE')
