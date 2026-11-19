"""Quick checks of funnel.py on a coarse grid (a few seconds). Run: python check.py
1. no-control evaluation matches the closed form P(|x + sigma B_T| <= w)
2. the HJB value equals the Feynman-Kac value of its own policy
3. Monte Carlo agrees with Feynman-Kac
4. shape of the optimal gap for one multiplier"""
import time
from dataclasses import replace
import numpy as np
from scipy.stats import norm
import funnel as F

t0 = time.time()
p = replace(F.P(), nx=1001, nt=1500)
x, s = F.grids(p)
U, Fr = F.evaluate(p, np.zeros((len(s), len(x))))
exact = 1 - (norm.cdf((p.w - x) / (p.sigma * np.sqrt(p.T))) - norm.cdf((-p.w - x) / (p.sigma * np.sqrt(p.T))))
print('1. free reject max err', float(np.abs(Fr - exact).max()))
c = 0.02
x, s, v, A = F.solve_hjb(p, c)
U, Fr = F.evaluate(p, A)
print('2. value vs policy eval max err', float(np.abs(v[0] - ((1 - Fr) - c * U)).max()))
r = F.summary(p, A); print('   entry avg', r)
print('3. MC', F.monte_carlo(p, A, n=20000, sub=2))
g = F.gap_of(p, A)
for q in (0, 0.25, 0.5, 0.75, 0.9, 0.97, 1.0):
    n = int(np.argmin(abs(s - q)))
    pushed = np.flatnonzero(A[n] != 0)
    print(f'4. s={q:.2f} gap +{g[n,0]:.3f} -{g[n,1]:.3f}  outer push extent', (x[pushed].min(), x[pushed].max()) if len(pushed) else None)
print('elapsed', round(time.time() - t0, 1))
