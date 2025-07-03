import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import os

# === SETTINGS ===
COUNTRY = "US"
WF_PATH = r"C:\Users\Mega-Pc\Desktop\Timeseries\AnxietyTimeseriesDashboard\blendedoutput\US\walkforward_garch_xgb_blend.csv"

if __name__ == "__main__":
    wf_df = pd.read_csv(WF_PATH, index_col=0, parse_dates=True)
    actual = wf_df['actual'].values
    blend_pred = wf_df['blend_pred'].values
    dates = wf_df.index

    # === BAYESIAN REGRESSION: blend_pred -> actual ===
    with pm.Model() as model:
        sigma = pm.HalfNormal('sigma', 10)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
        mu = beta[0] + beta[1] * blend_pred
        obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=actual)
        trace = pm.sample(2000, tune=1000, target_accept=0.95, random_seed=42, progressbar=True)

        # Posterior predictive sampling (returns ArviZ InferenceData)
        posterior_pred = pm.sample_posterior_predictive(trace, model=model, var_names=["obs"], progressbar=True)

    # --- Grab samples from InferenceData ---
    post_pred_samples = posterior_pred.posterior_predictive['obs'].values  # shape: (chains, draws, n_points)
    if post_pred_samples.ndim == 3:
        # Flatten chain and draw dimensions
        post_pred_samples = post_pred_samples.reshape(-1, post_pred_samples.shape[-1])  # (n_samples, n_points)

    # --- Compute Credible Intervals ---
    mean_pred = np.mean(post_pred_samples, axis=0)
    lower_ci = np.percentile(post_pred_samples, 2.5, axis=0)
    upper_ci = np.percentile(post_pred_samples, 97.5, axis=0)

    # --- Coverage ---
    in_interval = (actual >= lower_ci) & (actual <= upper_ci)
    coverage = np.mean(in_interval) * 100
    print(f"Bayesian 95% CI Coverage: {coverage:.2f}%")

    # --- Plot ---
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual, 'o-', label='Actual')
    plt.plot(dates, blend_pred, '--', label='Blend Prediction', color='orange')
    plt.plot(dates, mean_pred, '-', label='Bayesian Mean', color='green')
    plt.fill_between(dates, lower_ci, upper_ci, color='orange', alpha=0.22, label='Bayesian 95% CI')
    plt.title(f"Bayesian Interval for {COUNTRY} (Blend Forecast)\nCoverage: {coverage:.1f}%")
    plt.xlabel("Date")
    plt.ylabel("Anxiety Score")
    plt.legend()
    plt.tight_layout()
    plt.show()
