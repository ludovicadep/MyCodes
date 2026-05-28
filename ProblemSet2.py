
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import brentq


np.random.seed(100)   # change this for different runs
n = 2000              # change this: try small sample sizes!

mu  = np.random.normal(0, 1, n)
y   = np.random.normal(0, 1, n) + mu
tau = np.random.choice([-1, 1], size=n, p=[0.5, 0.5])

# different forecasts

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

# PIT Histogram preparation

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

#pit calibration

print("=" * 55)
print("KS tests — PIT vs Uniform (probabilistic calibration)")
print("=" * 55)
for name, F in forecasts.items():
    pit = F(y)
    ks = stats.kstest(pit, "uniform")
    sig = "*" if ks.pvalue < 0.05 else ""
    print(f"{name:<20}  D={ks.statistic:.4f}  p={ks.pvalue:.4f} {sig}")

# marginal calibration
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

########################## it does not work here check !!
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

