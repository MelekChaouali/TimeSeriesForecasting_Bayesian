import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import os

OUT_DIR = r"C:\Users\Mega-Pc\Desktop\Timeseries\AnxietyTimeseriesDashboard\blendedoutput"

# --- Utility: Get all country codes that have a walkforward CSV
def get_countries(out_dir):
    countries = []
    for name in os.listdir(out_dir):
        subdir = os.path.join(out_dir, name)
        if os.path.isdir(subdir):
            wf_path = os.path.join(subdir, "walkforward_garch_xgb_blend.csv")
            if os.path.exists(wf_path):
                countries.append(name)
    return sorted(countries)

def run_bayesian_intervals(country, out_dir=OUT_DIR, save_plots=True):
    wf_path = os.path.join(out_dir, country, "walkforward_garch_xgb_blend.csv")
    if not os.path.exists(wf_path):
        print(f"Skipping {country}: file not found")
        return None

    wf_df = pd.read_csv(wf_path, index_col=0, parse_dates=True)
    actual = wf_df['actual'].values
    blend_pred = wf_df['blend_pred'].values
    dates = wf_df.index

    # === 1. PRIOR ON MEAN ===
    with pm.Model() as model_prior_mean:
        mu_prior = pm.Normal('mu_prior', mu=blend_pred, sigma=2.5, shape=blend_pred.shape[0])
        sigma = pm.HalfNormal('sigma', 10)
        obs = pm.Normal('obs', mu=mu_prior, sigma=sigma, observed=actual)
        trace_mean = pm.sample(2000, tune=1000, target_accept=0.95, random_seed=42, progressbar=False)
        posterior_pred_mean = pm.sample_posterior_predictive(trace_mean, model=model_prior_mean, var_names=["obs"], progressbar=False)

    samples_mean = posterior_pred_mean.posterior_predictive['obs'].values
    if samples_mean.ndim == 3:
        samples_mean = samples_mean.reshape(-1, samples_mean.shape[-1])

    mean_pred1 = np.mean(samples_mean, axis=0)
    lower_ci1 = np.percentile(samples_mean, 2.5, axis=0)
    upper_ci1 = np.percentile(samples_mean, 97.5, axis=0)
    coverage1 = np.mean((actual >= lower_ci1) & (actual <= upper_ci1)) * 100

    # === 2. PRIOR ON REGRESSION COEFFICIENT ===
    with pm.Model() as model_prior_regression:
        beta0 = pm.Normal('beta0', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=1, sigma=0.25)
        sigma = pm.HalfNormal('sigma', 10)
        mu = beta0 + beta1 * blend_pred
        obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=actual)
        trace_reg = pm.sample(2000, tune=1000, target_accept=0.95, random_seed=42, progressbar=False)
        posterior_pred_reg = pm.sample_posterior_predictive(trace_reg, model=model_prior_regression, var_names=["obs"], progressbar=False)

    samples_reg = posterior_pred_reg.posterior_predictive['obs'].values
    if samples_reg.ndim == 3:
        samples_reg = samples_reg.reshape(-1, samples_reg.shape[-1])

    mean_pred2 = np.mean(samples_reg, axis=0)
    lower_ci2 = np.percentile(samples_reg, 2.5, axis=0)
    upper_ci2 = np.percentile(samples_reg, 97.5, axis=0)
    coverage2 = np.mean((actual >= lower_ci2) & (actual <= upper_ci2)) * 100

    # === PLOT AND SAVE ===
    fig, axs = plt.subplots(2, 1, figsize=(15, 11), sharex=True)

    axs[0].plot(dates, actual, 'o-', label='Actual')
    axs[0].plot(dates, blend_pred, '--', label='Blend Prediction', color='orange')
    axs[0].plot(dates, mean_pred1, '-', label='Bayesian Mean', color='green')
    axs[0].fill_between(dates, lower_ci1, upper_ci1, color='orange', alpha=0.22, label='Bayesian 95% CI')
    axs[0].set_title(f"{country} - Prior on Mean (Blend as Prior)\nCoverage: {coverage1:.1f}%")
    axs[0].legend()
    axs[0].set_ylabel("Anxiety Score")

    axs[1].plot(dates, actual, 'o-', label='Actual')
    axs[1].plot(dates, blend_pred, '--', label='Blend Prediction', color='orange')
    axs[1].plot(dates, mean_pred2, '-', label='Bayesian Mean', color='green')
    axs[1].fill_between(dates, lower_ci2, upper_ci2, color='orange', alpha=0.22, label='Bayesian 95% CI')
    axs[1].set_title(f"{country} - Prior on Regression, beta1~N(1,0.25)\nCoverage: {coverage2:.1f}%")
    axs[1].legend()
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel("Anxiety Score")

    plt.tight_layout()
    if save_plots:
        save_path = os.path.join(out_dir, country, f"bayesian_cis_{country}.png")
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    plt.close(fig)

    return {
        "country": country,
        "coverage_prior_mean": coverage1,
        "coverage_prior_regression": coverage2
    }

# === MAIN LOOP ===
if __name__ == "__main__":
    countries = get_countries(OUT_DIR)
    print(f"Found {len(countries)} countries: {countries}")

    results = []
    for c in countries:
        print(f"\n--- Running Bayesian intervals for: {c} ---")
        res = run_bayesian_intervals(c, OUT_DIR)
        if res: results.append(res)

    # --- Print summary ---
    print("\n=== Summary of Coverages ===")
    df = pd.DataFrame(results)
    print(df)
    # Optionally: df.to_csv("bayesian_interval_coverages.csv", index=False)
