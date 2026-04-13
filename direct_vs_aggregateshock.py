"""
direct_vs_aggregateshock.py
============================
Corrected translation of the MATLAB script `direct_vs_aggregateshock`.

Bugs fixed
----------
1. Variable naming conflict: `i` used as both the inflation array and loop
   index → renamed inflation to `pi_rate`.
2. Dimension error in X_sh: `Omega_direct @ eps[…]` (5×5 × Teff×5 invalid)
   → corrected to `eps[…] @ Omega_direct` (Teff×5 × 5×5 = Teff×5).
3. `IRF_dir` / `IRF_dir_se` had wrong 2nd dimension K instead of K+1 (5 shocks).
4. Direct LP index `b[1:K+1]` extracted only K=4 coefficients → fixed to
   `b[1:K+2]` for K+1=5 shocks.
5. `aggr_hat = eps @ B.T` where B = A0_true @ Omega_true (2×5) gave T×2 instead
   of T×5 → corrected to `aggr_hat = eps @ Omega_true.T` (T×5).
6. `A_h = zeros(N, K)` → corrected to `zeros(N, K+1)` for 5 aggregate shocks.
7. Aggregate LP index `b[1:K+1]` → corrected to `b[1:K+2]`.
8. `Omega_est` was never defined → replaced with `Omega_true`.
9. DGP did not incorporate shocks into Y (the simulation loop used plain AR(1)
   without the aggregate shocks), making LP estimates meaningless → added
   `A0_true @ aggr[t]` to both outcome equations.
10. Missing plotting code → added subplot-based IRF comparison figure.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Parameters ────────────────────────────────────────────────────────────────
T   = 600          # sample size
K   = 4            # number of structural shocks (crisis added separately)
N   = 2            # number of outcomes (log IP, unemployment)
H   = 20           # max horizon
p   = 4            # lag order for LP controls
rho = 0.70         # AR(1) persistence for true IRF decay
shock_names   = ['FP', 'MP', 'TFP', 'Oil', 'Crisis']
outcome_names = ['Unemployment', 'Inflation']

# ── True parameters ───────────────────────────────────────────────────────────
Omega_true = np.array([
    [0.80, 0.20, 0.10, 0.30, 0.00],
    [0.10, 0.90, 0.20, 0.10, 0.30],
    [0.20, 0.10, 0.70, 0.20, 0.10],
    [0.30, 0.20, 0.10, 0.80, 0.20],
    [0.00, 0.10, 0.20, 0.10, 0.90],
])  # 5×5

A0_true = np.array([
    [ 0.50, -0.30,  0.40, -0.20, -0.50],   # unemployment responses (1×5)
    [-0.30,  0.20, -0.20,  0.30,  0.40],   # inflation responses    (1×5)
])  # 2×5

# True IRF(h) = rho^h * A0_true @ Omega_true  →  2×5 at each h
true_irf = np.zeros((N, K + 1, H + 1))
for h in range(H + 1):
    true_irf[:, :, h] = (rho ** h) * A0_true @ Omega_true

# ── DGP: simulate shocks and outcomes ────────────────────────────────────────
rng1 = np.random.default_rng(42)
crises_shock        = np.zeros(T)
crises_shock[T - 1] = 1.0                         # one-time shock at end
eps_rand = rng1.standard_normal((T, K))
eps      = np.column_stack([eps_rand, crises_shock])   # T×(K+1)
aggr     = eps @ Omega_true.T                          # T×5

# Re-seed for the outcome simulation (mirrors MATLAB's rng(42) reset)
rng2 = np.random.default_rng(42)

# Bug 9 fix: Y simulation now incorporates aggregate shocks
unemp = np.zeros(T)   # unemployment rate
infl  = np.zeros(T)   # inflation rate
unemp[0] = 6.0
infl[0]  = 2.0

for t in range(1, T):
    unemp[t] = (0.5 + 0.9 * unemp[t - 1]
                + A0_true[0, :] @ aggr[t]     # Bug 9: add shock effect
                + 0.5 * rng2.standard_normal())
    infl[t]  = (0.5 + 0.8 * infl[t - 1]
                + A0_true[1, :] @ aggr[t]     # Bug 9: add shock effect
                + 0.4 * rng2.standard_normal())

Y = np.column_stack([unemp, infl])   # T×2

# Omega_direct: diagonal matrix using diagonal of Omega_true
Omega_direct = np.diag(np.diag(Omega_true))   # 5×5 diagonal

# ── OLS helper: returns (coefficients, HC0 SE) ───────────────────────────────
def ols(X, y):
    """OLS with heteroskedasticity-robust (HC0) standard errors.

    Uses lstsq for robustness when columns are (near-)singular
    (e.g. the crisis-shock column is all-zeros for h >= 1).
    The HC0 meat is computed as (X*e).T @ (X*e) to avoid forming a
    large Teff×Teff diagonal matrix.
    """
    b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid      = y - X @ b
    # (X'X)^{-1}  – use pinv to handle rank-deficient designs
    XtX_inv = np.linalg.pinv(X.T @ X)
    # HC0 meat: sum_t e_t^2 * x_t x_t' = (X diag(e))' @ (X diag(e))
    Xe   = X * resid[:, None]   # Teff × ncols
    meat = Xe.T @ Xe
    vcov = XtX_inv @ meat @ XtX_inv
    se   = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    return b, se

# ── Direct LP ─────────────────────────────────────────────────────────────────
# Bug 3 fix: shape uses K+1 (5 shocks) instead of K (4)
IRF_dir    = np.zeros((N, K + 1, H + 1))
IRF_dir_se = np.zeros((N, K + 1, H + 1))

for h in range(H + 1):
    t0   = p           # 0-based: row index p (= 5th row in 1-based)
    t1   = T - h - 1   # inclusive last index (0-based)
    Teff = t1 - t0 + 1

    dep = Y[t0 + h: t1 + h + 1, :] - Y[t0 - 1: t1, :]   # Teff×N

    # Bug 2 fix: eps[…] @ Omega_direct  (Teff×5 @ 5×5 = Teff×5)
    X_sh   = eps[t0: t1 + 1, :] @ Omega_direct            # Teff×(K+1)
    X_ctrl = np.zeros((Teff, N * p))
    for l in range(1, p + 1):
        X_ctrl[:, (l - 1) * N: l * N] = Y[t0 - l: t1 - l + 1, :]
    X = np.column_stack([np.ones(Teff), X_sh, X_ctrl])    # Teff×(1+(K+1)+N*p)

    for j in range(N):
        b, se = ols(X, dep[:, j])
        # Bug 4 fix: b[1:K+2] extracts K+1=5 coefficients
        IRF_dir[j, :, h]    = b[1: K + 2]
        IRF_dir_se[j, :, h] = se[1: K + 2]

# ── Aggregate LP ──────────────────────────────────────────────────────────────
# Bug 5 fix: use full Omega_true (not B = A0_true @ Omega_true) so aggr_hat is T×5
aggr_hat = eps @ Omega_true.T    # T×(K+1)

# Bug 6 fix: shape uses K+1 (5 shocks)
IRF_agg   = np.zeros((N, K + 1, H + 1))
A_h_store = np.zeros((H + 1, N, K + 1))

for h in range(H + 1):
    t0   = p
    t1   = T - h - 1
    Teff = t1 - t0 + 1

    dep    = Y[t0 + h: t1 + h + 1, :] - Y[t0 - 1: t1, :]
    X_ag   = aggr_hat[t0: t1 + 1, :]                       # Teff×(K+1)
    X_ctrl = np.zeros((Teff, N * p))
    for l in range(1, p + 1):
        X_ctrl[:, (l - 1) * N: l * N] = Y[t0 - l: t1 - l + 1, :]
    X = np.column_stack([np.ones(Teff), X_ag, X_ctrl])

    # Bug 6 fix: A_h is N×(K+1)
    A_h = np.zeros((N, K + 1))
    for j in range(N):
        b = np.linalg.lstsq(X, dep[:, j], rcond=None)[0]
        # Bug 7 fix: b[1:K+2] for K+1=5 coefficients
        A_h[j, :] = b[1: K + 2]
    A_h_store[h, :, :] = A_h
    # Bug 8 fix: Omega_est → Omega_true
    IRF_agg[:, :, h] = A_h @ Omega_true

# ── Plot IRFs ─────────────────────────────────────────────────────────────────
horizons  = np.arange(H + 1)
ci_factor = 1.96   # 95 % CI

fig, axes = plt.subplots(N, K + 1, figsize=(18, 7), sharey='row')
fig.suptitle('IRF: True vs Direct LP vs Aggregate LP', fontsize=14, fontweight='bold')

for n in range(N):
    for k in range(K + 1):
        ax = axes[n, k]

        true_h = true_irf[n, k, :]
        dir_h  = IRF_dir[n, k, :]
        dir_se = IRF_dir_se[n, k, :]
        agg_h  = IRF_agg[n, k, :]

        # True IRF
        ax.plot(horizons, true_h, 'k-',  lw=2,   label='True IRF')
        # Direct LP + shaded 95 % CI
        ax.plot(horizons, dir_h,  'b--', lw=1.5, label='Direct LP')
        ax.fill_between(horizons,
                        dir_h - ci_factor * dir_se,
                        dir_h + ci_factor * dir_se,
                        color='blue', alpha=0.15)
        # Aggregate LP
        ax.plot(horizons, agg_h,  'r-.',  lw=1.5, label='Aggregate LP')

        ax.axhline(0, color='k', lw=0.7, ls=':')
        ax.set_xlim(0, H)
        ax.grid(True, lw=0.4)

        if n == 0:
            ax.set_title(shock_names[k], fontweight='bold')
        if k == 0:
            ax.set_ylabel(outcome_names[n])
        if n == N - 1:
            ax.set_xlabel('Horizon')
        if n == N - 1 and k == (K + 1) // 2:
            ax.legend(fontsize=7, loc='best')

plt.tight_layout()
out_path = 'irf_direct_vs_aggregate.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'IRF plot saved to {out_path}')
