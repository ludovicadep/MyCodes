"""
Simulation study: probabilistic forecast evaluation
Converted from R to Python
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import brentq

# ─── Simulation Setup ────────────────────────────────────────────────────────

np.random.seed(100)   # change this for different runs
n = 2000              # change this: try small sample sizes!

mu  = np.random.normal(0, 1, n)
y   = np.random.normal(0, 1, n) + mu
tau = np.random.choice([-1, 1], size=n, p=[0.5, 0.5])

# ─── (a) Forecast CDFs ───────────────────────────────────────────────────────

def F_perf(x):  return stats.norm.cdf(x - mu)
def F_clim(x):  return stats.norm.cdf(x / np.sqrt(2))
def F_unf(x):   return 0.5 * (stats.norm.cdf(x - mu) + stats.norm.cdf(x - mu - tau))
def F_sign(x):  return stats.norm.cdf(x + mu)

sd_under = 3/4
def F_under(x): return stats.norm.cdf((x - mu) / sd_under)

sd_over = 5/4
def F_over(x):  return stats.norm.cdf((x - mu) / sd_over)

pbias =  0.3
nbias = -0.3
def F_pbias(x): return stats.norm.cdf(x - (mu + pbias))
def F_nbias(x): return stats.norm.cdf(x - (mu + nbias))

forecasts = {
    "Perfect":          F_perf,
    "Climatological":   F_clim,
    "Unfocused":        F_unf,
    "Sign-reversed":    F_sign,
    "Overdispersed":    F_over,
    "Underdispersed":   F_under,
    "Overprediction":   F_pbias,
    "Underprediction":  F_nbias,
}

# ─── (b) PIT Histograms & Marginal Calibration ───────────────────────────────

n_bins = 10

# --- PIT histograms ---
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle("PIT Histograms", fontsize=14, fontweight="bold")

for ax, (name, F) in zip(axes.flat, forecasts.items()):
    pit = F(y)
    ax.hist(pit, bins=n_bins, edgecolor="white", color="steelblue")
    ax.axhline(n / n_bins, color="red", linestyle="--", linewidth=1.2, label="Uniform")
    ax.set_title(name)
    ax.set_xlabel("PIT")
    ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig("pit_histograms.png", dpi=150)
plt.show()

# --- KS tests for probabilistic calibration ---
print("=" * 55)
print("KS tests — PIT vs Uniform (probabilistic calibration)")
print("=" * 55)
for name, F in forecasts.items():
    pit = F(y)
    ks = stats.kstest(pit, "uniform")
    sig = "*" if ks.pvalue < 0.05 else ""
    print(f"{name:<20}  D={ks.statistic:.4f}  p={ks.pvalue:.4f} {sig}")

# --- Marginal calibration ---
F_sample = stats.ecdf(y)   # empirical CDF of the observations

def avg_cdf(F, x_arr):
    """Vectorised mean of a forecast CDF over the sample."""
    return np.array([np.mean(F(x)) for x in x_arr])

x_seq = np.linspace(-10, 10, 2001)

# PP plots
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle("PP Plots (Marginal Calibration)", fontsize=14, fontweight="bold")

for ax, (name, F) in zip(axes.flat, forecasts.items()):
    fcast_avg = avg_cdf(F, x_seq)
    obs_ecdf   = F_sample.cdf.evaluate(x_seq)
    ax.plot(fcast_avg, obs_ecdf, color="steelblue", linewidth=1.5)
    ax.plot([0, 1], [0, 1], "r--", linewidth=1)
    ax.set_title(name)
    ax.set_xlabel("Avg forecast CDF")
    ax.set_ylabel("Empirical CDF")

plt.tight_layout()
plt.savefig("pp_plots.png", dpi=150)
plt.show()

# Difference plots
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle("Marginal Calibration — Difference Plots", fontsize=14, fontweight="bold")

x_seq2 = np.linspace(-5, 5, 1001)

for ax, (name, F) in zip(axes.flat, forecasts.items()):
    fcast_avg = avg_cdf(F, x_seq2)
    obs_ecdf   = F_sample.cdf.evaluate(x_seq2)
    ax.plot(x_seq2, fcast_avg - obs_ecdf, color="steelblue", linewidth=1.5)
    ax.axhline(0, color="red", linestyle="--", linewidth=1)
    ax.set_ylim(-0.1, 0.1)
    ax.set_title(name)
    ax.set_xlabel("x")
    ax.set_ylabel("Avg CDF − Empirical CDF")

plt.tight_layout()
plt.savefig("diff_plots.png", dpi=150)
plt.show()

# KS tests for marginal calibration
print("\n" + "=" * 55)
print("KS tests — observations vs average forecast CDF")
print("  (marginal calibration)")
print("=" * 55)

for name, F in forecasts.items():
    f_avg = lambda x, _F=F: avg_cdf(_F, np.atleast_1d(x)).squeeze()
    ks = stats.kstest(y, f_avg)
    sig = "*" if ks.pvalue < 0.05 else ""
    print(f"{name:<20}  D={ks.statistic:.4f}  p={ks.pvalue:.4f} {sig}")

# ─── (c) Sharpness ───────────────────────────────────────────────────────────

def qmixnorm(p):
    """Quantile of N(0,1)/N(1,1) equal-weight mixture."""
    cdf = lambda x: 0.5 * (stats.norm.cdf(x) + stats.norm.cdf(x - 1))
    return brentq(lambda x: cdf(x) - p,
                  stats.norm.ppf(p) - 1,
                  stats.norm.ppf(p) + 1)

w50_std  = stats.norm.ppf(0.75) - stats.norm.ppf(0.25)
w50_unf  = qmixnorm(0.75) - qmixnorm(0.25)
w90_std  = stats.norm.ppf(0.95) - stats.norm.ppf(0.05)
w90_unf  = qmixnorm(0.95) - qmixnorm(0.05)

forecast_names = list(forecasts.keys())
width50 = [w50_std, np.sqrt(2)*w50_std, w50_unf, w50_std,
           sd_over*w50_std, sd_under*w50_std, w50_std, w50_std]
width90 = [w90_std, np.sqrt(2)*w90_std, w90_unf, w90_std,
           sd_over*w90_std, sd_under*w90_std, w90_std, w90_std]

print("\n" + "=" * 45)
print("Sharpness — central prediction interval widths")
print(f"{'Forecast':<20} {'50%':>8} {'90%':>8}")
print("-" * 45)
for name, w50, w90 in zip(forecast_names, width50, width90):
    print(f"{name:<20} {w50:>8.2f} {w90:>8.2f}")

# ─── (d) Proper Scoring Rules ────────────────────────────────────────────────

# ── CRPS for normal distributions ──────────────────────────────────────────
def crps_norm(y, mu, sigma):
    """CRPS for N(mu, sigma^2)."""
    z = (y - mu) / sigma
    return sigma * (z * (2*stats.norm.cdf(z) - 1)
                    + 2*stats.norm.pdf(z)
                    - 1/np.sqrt(np.pi))

def crps_mixnorm(y, mus, sigmas, weights=None):
    """
    CRPS for an equal-weight mixture of K normals.
    mus, sigmas : arrays of shape (n, K)
    """
    n_obs, K = mus.shape
    if weights is None:
        weights = np.ones(K) / K

    # Term 1: E[|X - y|]
    term1 = np.zeros(n_obs)
    for k in range(K):
        term1 += weights[k] * crps_norm(y, mus[:, k], sigmas[:, k])

    # Term 2: -0.5 * E[|X - X'|] for mixture
    # For two normals i,j: E[|X_i - X_j|] = analytical formula
    term2 = np.zeros(n_obs)
    for i in range(K):
        for j in range(K):
            mu_diff  = mus[:, i] - mus[:, j]
            sig_comb = np.sqrt(sigmas[:, i]**2 + sigmas[:, j]**2)
            z        = mu_diff / sig_comb
            e_abs    = sig_comb * (2*stats.norm.pdf(z) + 2*z*stats.norm.cdf(z) - z)
            term2   += weights[i] * weights[j] * e_abs

    # CRPS = E|X-y| - 0.5*E|X-X'|
    # term1 already contains CRPS for each component individually;
    # we need the full formula directly:
    # CRPS(mix, y) = sum_k w_k CRPS(F_k, y)
    #              + 0.5 * sum_{i,j} w_i w_j E|X_i - X_j| (cross terms with same obs cancel)
    # Simpler: use the energy score formula below.
    return term1 - 0.5 * term2 + 0.5 * term2  # simplifies – use direct formula

def crps_mixnorm_direct(y, mus, sigmas):
    """
    Direct CRPS for equal-weight two-component mixture.
    Formula: mean_k CRPS(N_k, y)
           - 1/(2K^2) sum_{i,j} E|X_i - X_j|
    """
    n_obs, K = mus.shape
    w = 1.0 / K
    # E|X - y| part
    energy_y = np.zeros(n_obs)
    for k in range(K):
        energy_y += w * crps_norm(y, mus[:, k], sigmas[:, k])

    # E|X - X'| part
    energy_xx = np.zeros(n_obs)
    for i in range(K):
        for j in range(K):
            mu_diff  = mus[:, i] - mus[:, j]
            sig_ij   = np.sqrt(sigmas[:, i]**2 + sigmas[:, j]**2)
            z        = mu_diff / sig_ij
            energy_xx += w * w * sig_ij * (
                2*stats.norm.pdf(z) + 2*z*stats.norm.cdf(z) - z
            )
    # Oops – need proper formula. Use the known decomposition:
    # CRPS(F, y) = E_F|X - y| - 0.5 E_F|X - X'|
    # E_F|X-y| = sum_k w_k E_{N_k}|X-y|  (linearity)
    # E_F|X-X'| = sum_{i,j} w_i w_j E|X_i - X_j'|
    e_xy = np.zeros(n_obs)
    for k in range(K):
        z_k   = (y - mus[:, k]) / sigmas[:, k]
        e_xy += w * sigmas[:, k] * (
            2*stats.norm.pdf(z_k) + z_k*(2*stats.norm.cdf(z_k) - 1)
        )

    e_xx = np.zeros(n_obs)
    for i in range(K):
        for j in range(K):
            mu_diff = mus[:, i] - mus[:, j]
            sig_ij  = np.sqrt(sigmas[:, i]**2 + sigmas[:, j]**2)
            z       = np.abs(mu_diff) / sig_ij
            e_xx   += w * w * sig_ij * (
                2*stats.norm.pdf(z) + 2*z*stats.norm.cdf(z) - z
            )

    return e_xy - 0.5 * e_xx


# ── Logarithmic score ───────────────────────────────────────────────────────
def logs_norm(y, mu, sigma):
    return -stats.norm.logpdf(y, mu, sigma)

def logs_mixnorm(y, mus, sigmas):
    n_obs, K = mus.shape
    density = np.zeros(n_obs)
    for k in range(K):
        density += (1/K) * stats.norm.pdf(y, mus[:, k], sigmas[:, k])
    return -np.log(density)

# ── Hyvärinen score ─────────────────────────────────────────────────────────
def d1norm(x, m, s):
    return stats.norm.pdf(x, m, s) * (m - x) / s**2

def d2norm(x, m, s):
    return stats.norm.pdf(x, m, s) * ((x - m)**2 / s**4 - 1/s**2)

def mean_hyv(f, f1, f2):
    fv  = f(y)
    f1v = f1(y)
    f2v = f2(y)
    return np.mean(2*f2v/fv - (f1v/fv)**2)

# Densities and derivatives
def make_norm_fns(loc, scale):
    f  = lambda x: stats.norm.pdf(x, loc, scale)
    f1 = lambda x: d1norm(x, loc, scale)
    f2 = lambda x: d2norm(x, loc, scale)
    return f, f1, f2

def make_mixnorm_fns(loc1, loc2, scale=1.0):
    f  = lambda x: 0.5*(stats.norm.pdf(x, loc1, scale) + stats.norm.pdf(x, loc2, scale))
    f1 = lambda x: 0.5*(d1norm(x, loc1, scale) + d1norm(x, loc2, scale))
    f2 = lambda x: 0.5*(d2norm(x, loc1, scale) + d2norm(x, loc2, scale))
    return f, f1, f2

f_perf,  f1_perf,  f2_perf  = make_norm_fns(mu, 1)
f_clim,  f1_clim,  f2_clim  = make_norm_fns(0,  np.sqrt(2))
f_unf,   f1_unf,   f2_unf   = make_mixnorm_fns(mu, mu + tau)
f_sign,  f1_sign,  f2_sign  = make_norm_fns(-mu, 1)
f_over,  f1_over,  f2_over  = make_norm_fns(mu, sd_over)
f_under, f1_under, f2_under = make_norm_fns(mu, sd_under)
f_pbias, f1_pbias, f2_pbias = make_norm_fns(mu + pbias, 1)
f_nbias, f1_nbias, f2_nbias = make_norm_fns(mu + nbias, 1)

# Mixture matrices for CRPS / LogS
mus2   = np.column_stack([mu, mu + tau])
sigs2  = np.ones((n, 2))

# ── Compute all scores ───────────────────────────────────────────────────────
scores = {
    "Perfect":        (crps_norm(y, mu, 1),
                       logs_norm(y, mu, 1),
                       mean_hyv(f_perf, f1_perf, f2_perf)),
    "Climatological": (crps_norm(y, 0, np.sqrt(2)),
                       logs_norm(y, 0, np.sqrt(2)),
                       mean_hyv(f_clim, f1_clim, f2_clim)),
    "Unfocused":      (crps_mixnorm_direct(y, mus2, sigs2),
                       logs_mixnorm(y, mus2, sigs2),
                       mean_hyv(f_unf, f1_unf, f2_unf)),
    "Sign-reversed":  (crps_norm(y, -mu, 1),
                       logs_norm(y, -mu, 1),
                       mean_hyv(f_sign, f1_sign, f2_sign)),
    "Overdispersed":  (crps_norm(y, mu, sd_over),
                       logs_norm(y, mu, sd_over),
                       mean_hyv(f_over, f1_over, f2_over)),
    "Underdispersed": (crps_norm(y, mu, sd_under),
                       logs_norm(y, mu, sd_under),
                       mean_hyv(f_under, f1_under, f2_under)),
    "Overprediction": (crps_norm(y, mu + pbias, 1),
                       logs_norm(y, mu + pbias, 1),
                       mean_hyv(f_pbias, f1_pbias, f2_pbias)),
    "Underprediction":(crps_norm(y, mu + nbias, 1),
                       logs_norm(y, mu + nbias, 1),
                       mean_hyv(f_nbias, f1_nbias, f2_nbias)),
}

print("\n" + "=" * 55)
print(f"{'Forecast':<20} {'CRPS':>8} {'LogS':>8} {'HyvS':>8}")
print("-" * 55)
for name, (crps_v, logs_v, hyv_v) in scores.items():
    print(f"{name:<20} {np.mean(crps_v):>8.3f} {np.mean(logs_v):>8.3f} {hyv_v:>8.3f}")

print("\nPlots saved: pit_histograms.png, pp_plots.png, diff_plots.png")