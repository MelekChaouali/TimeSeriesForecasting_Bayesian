import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
import optuna

# === Logging setup ===
LOG_PATH = r"C:\Users\Mega-Pc\Desktop\Timeseries\sarimax_garch_output\metrics.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_PATH,
    filemode='w',  # Overwrite each run
    format='[%(asctime)s] %(levelname)s %(message)s',
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

# Settings
DATA_PATH = r"C:\Users\Mega-Pc\Desktop\Timeseries\mental_health_trends.csv"
MOBILITY_DIR = r"C:\Users\Mega-Pc\Desktop\Timeseries"
OUT_DIR = os.path.join(MOBILITY_DIR, "sarimax_garch_output")
FOLDS, HORIZON = 4, 12
EPS = 1e-3
JOBS = 1  # For parallel later
FORECAST_MONTHS = 60  # 5 years ahead

os.makedirs(OUT_DIR, exist_ok=True)

def get_available_countries_mobility(data_dir):
    files = os.listdir(data_dir)
    countries = []
    for f in files:
        if f.startswith('google_mobility_') and f.endswith('.csv'):
            code = f.split('_')[-1].replace('.csv','')
            countries.append(code)
    return countries

OECD_CODES = get_available_countries_mobility(MOBILITY_DIR)
logging.info(f"Countries with mobility data: {OECD_CODES}")

class DataPipeline:
    def __init__(self, mobility_csv, country_code):
        self.mobility_csv = mobility_csv
        self.country = country_code

    @staticmethod
    def to_logit(x):
        clipped = np.clip(x, 0.1, 99.9)
        p = (clipped + EPS) / (100 + 2*EPS)
        return np.log(p/(1-p))

    @staticmethod
    def from_logit(z):
        clipped = np.clip(z, -10, 10)
        p = 1/(1+np.exp(-clipped))
        return p*(100+2*EPS)-EPS

    @staticmethod
    def dedup(s):
        return s[~s.index.duplicated()]

    def load_target(self, path):
        df = pd.read_csv(path, parse_dates=['date']).set_index('date').asfreq('W')
        if 'iso_code' in df.columns:
            df = df[df['iso_code'] == self.country]
        raw = self.dedup(df['anxiety'].resample('MS').mean().dropna())
        logit = self.to_logit(raw)
        return raw, logit

    def load_exog(self, path, col):
        df = pd.read_csv(path)
        if 'iso_code' in df.columns:
            df = df[df['iso_code'] == self.country]
        df = df.iloc[:, :2]
        if df.empty or df.shape[1] < 2:
            raise ValueError("No exog data found")
        df.columns = ['date', col]
        df['date'] = pd.to_datetime(df['date']) + pd.offsets.MonthBegin()
        df = df.drop_duplicates('date')
        df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
        return df.groupby('month')[col].mean()

    def build_exogenous(self, raw_index):
        mob = self.load_exog(self.mobility_csv, 'mob_work')
        exog = pd.DataFrame(mob)
        exog = exog.reindex(raw_index)
        exog = exog.interpolate().fillna(0)
        exog['exam_dummy'] = raw_index.month.isin([1,6,12]).astype(int)
        for c in exog.columns:
            std = exog[c].std()
            exog[c] = (exog[c] - exog[c].mean())/(std if std>0 else 1)
        return exog

    def prepare(self, path):
        raw, logit = self.load_target(path)
        exog = self.build_exogenous(raw.index)
        exog = exog.loc[logit.index]
        return raw, logit, exog

class ForecastPipeline:
    def __init__(self, order, seasonal_order):
        self.order = order
        self.seasonal_order = seasonal_order

    @staticmethod
    def fit_sarimax(y, X, order, seasonal_order):
        for m in ['lbfgs','powell','bfgs','nm']:
            try:
                mod = SARIMAX(y, exog=X, order=order, seasonal_order=seasonal_order,
                              enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(method=m, disp=False, maxiter=500)
                if not np.isnan(res.params).any():
                    return res
            except Exception as e:
                continue
        return None

    def walk_forward(self, logit, exog):
        preds, actuals, residuals = [], [], []
        for i in range(HORIZON):
            cut = len(logit) - HORIZON + i
            res = self.fit_sarimax(logit[:cut], exog[:cut], self.order, self.seasonal_order)
            if res is None:
                preds.append(np.nan)
                actuals.append(np.nan)
                residuals.append(np.nan)
                continue
            f = res.get_forecast(1, exog=exog[cut:cut+1])
            pred = f.predicted_mean.iloc[0]
            preds.append(pred)
            act = logit.iloc[cut]
            actuals.append(act)
            residuals.append(act - pred)
        idx = logit.index[-HORIZON:]
        df = pd.DataFrame({
            'actual': DataPipeline.from_logit(pd.Series(actuals,index=idx)),
            'forecast': DataPipeline.from_logit(pd.Series(preds,index=idx)),
            'logit_pred': pd.Series(preds, index=idx),
            'logit_actual': pd.Series(actuals, index=idx),
            'residuals': residuals
        }, index=idx)
        return df

    def garch_intervals(self, wf_df, alpha=0.05):
        resids = np.array(wf_df['logit_actual'] - wf_df['logit_pred'])
        resids = resids[~np.isnan(resids)]
        if len(resids) < 10:
            wf_df['garch_lower'] = np.nan
            wf_df['garch_upper'] = np.nan
            return wf_df
        garch_mod = arch_model(resids, p=1, q=1, mean='zero', vol='GARCH', dist='normal')
        garch_res = garch_mod.fit(disp='off')
        # Get the one-step-ahead predicted volatility for each
        forecast = garch_res.forecast(start=0, horizon=1)
        sigmas = forecast.variance.values.flatten()[:len(wf_df)]
        sigmas = np.sqrt(sigmas)
        z = 1.96  # 95%
        wf_df['garch_lower'] = DataPipeline.from_logit(wf_df['logit_pred'] - z * sigmas)
        wf_df['garch_upper'] = DataPipeline.from_logit(wf_df['logit_pred'] + z * sigmas)
        return wf_df

    def final_forecast(self, logit, exog, steps=60):
        # Use last 12 exog rows, repeat/average for future
        last_exog = exog.iloc[-12:]
        future_idx = pd.date_range(logit.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq='MS')
        # Take mean of each exog for simplicity (or you can repeat pattern)
        future_exog = pd.DataFrame({col: [last_exog[col].mean()]*steps for col in exog.columns}, index=future_idx)
        model_fit = self.fit_sarimax(logit, exog, self.order, self.seasonal_order)
        fc = model_fit.get_forecast(steps, exog=future_exog)
        fc_mean = DataPipeline.from_logit(fc.predicted_mean)
        fc_ci = fc.conf_int().apply(DataPipeline.from_logit)
        return fc_mean, fc_ci

def objective(trial, logit, exog):
    od = (trial.suggest_int('p',0,2), trial.suggest_int('d',0,1), trial.suggest_int('q',0,2))
    ss = (trial.suggest_int('P',0,1), trial.suggest_int('D',0,1), trial.suggest_int('Q',0,1), 12)
    if sum(od)+sum(ss[:-1])>4:
        return np.inf
    errs=[]; step=(len(logit)-HORIZON)//FOLDS
    for k in range(FOLDS):
        cut=min(len(logit)-HORIZON, step*(k+1))
        y1, y2 = logit[:cut], logit[cut:cut+HORIZON]
        X1, X2 = exog[:cut], exog[cut:cut+HORIZON]
        res = ForecastPipeline(od, ss).fit_sarimax(y1, X1, od, ss)
        if not res: continue
        f = res.get_forecast(len(y2), exog=X2).predicted_mean
        errs.append(mean_absolute_error(DataPipeline.from_logit(y2), DataPipeline.from_logit(f)))
    return np.mean(errs) if errs else np.inf

def run_for_country(country):
    try:
        country_dir = os.path.join(OUT_DIR, country)
        os.makedirs(country_dir, exist_ok=True)
        mob_csv = os.path.join(MOBILITY_DIR, f"google_mobility_{country}.csv")
        pipeline = DataPipeline(mob_csv, country)
        raw, logit, exog = pipeline.prepare(DATA_PATH)

        # Tune model
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: objective(t, logit, exog), timeout=180, n_trials=40)
        bp = study.best_params
        order = (bp['p'],bp['d'],bp['q'])
        seasonal = (bp['P'],bp['D'],bp['Q'],12)

        # Walk-forward eval
        forecaster = ForecastPipeline(order, seasonal)
        wf_df = forecaster.walk_forward(logit, exog)
        wf_df = forecaster.garch_intervals(wf_df)

        # Metrics for SARIMAX GARCH CIs
        def _cov(a, lo, hi): return np.mean((a >= lo) & (a <= hi))*100
        metrics = {
            'Country': country,
            'RMSE': mean_squared_error(wf_df['actual'], wf_df['forecast'], squared=False),
            'MAPE': np.mean(np.abs((wf_df['actual']-wf_df['forecast'])/wf_df['actual']))*100,
            'GARCH_Coverage': _cov(wf_df['actual'], wf_df['garch_lower'], wf_df['garch_upper']),
        }
        logging.info(f"{country} GARCH CI: {metrics}")

        # Save walk-forward CIs
        wf_df.to_csv(os.path.join(country_dir, 'walkforward_garch.csv'))

        # === Plotting (backtest) ===
        plt.figure(figsize=(10,4))
        plt.plot(raw, label='Obs')
        plt.plot(wf_df['forecast'], label='Forecast')
        plt.fill_between(wf_df.index, wf_df['garch_lower'], wf_df['garch_upper'], color='purple', alpha=0.18, label='GARCH CI')
        plt.legend(); plt.tight_layout()
        plt.title(f'{country} SARIMAX+GARCH Backtest CIs')
        plt.savefig(os.path.join(country_dir,'backtest_garch_cis.png'))
        plt.close()

        # === Final Forecast (5y out-of-sample) ===
        fc_mean, fc_ci = forecaster.final_forecast(logit, exog, steps=FORECAST_MONTHS)
        fc_mean.to_csv(os.path.join(country_dir, "forecast_5year.csv"))
        fc_ci.to_csv(os.path.join(country_dir, "forecast_5year_ci.csv"))

        # Plot final forecast
        plt.figure(figsize=(12,4))
        plt.plot(raw, label='Observed')
        plt.plot(fc_mean, label='5yr Forecast')
        plt.fill_between(fc_ci.index, fc_ci.iloc[:,0], fc_ci.iloc[:,1], alpha=0.2, label='95% CI')
        plt.legend()
        plt.title(f"{country} 5-year SARIMAX+GARCH Forecast")
        plt.tight_layout()
        plt.savefig(os.path.join(country_dir, 'forecast_5year.png'))
        plt.close()

    except Exception as e:
        logging.warning(f"{country}: {e}")

if __name__ == '__main__':
    for c in OECD_CODES:
        run_for_country(c)
