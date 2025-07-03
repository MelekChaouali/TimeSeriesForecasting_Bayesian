import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import optuna

# === Logging setup ===
LOG_PATH = r"C:\Users\Mega-Pc\Desktop\Timeseries\sarimax_bootstrap_output\metrics.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_PATH,
    filemode='w',  # 'w' to overwrite each run, 'a' to append
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
OUT_DIR = os.path.join(MOBILITY_DIR, "sarimax_bootstrap_output")
FOLDS, HORIZON = 4, 12
EPS = 1e-3
BOOTSTRAP_SAMPLES = 1000

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

    def bootstrap_intervals(self, wf_df, alpha=0.05, n_samples=1000):
        resids = np.array(wf_df['actual'] - wf_df['forecast'])
        n = len(wf_df)
        if n < 5:
            wf_df['boot_lower'] = np.nan
            wf_df['boot_upper'] = np.nan
            return wf_df
        boot_lo = []
        boot_hi = []
        for i in range(n):
            samples = np.random.choice(resids, size=n_samples, replace=True)
            pred = wf_df['forecast'].iloc[i]
            dist = pred + samples
            boot_lo.append(np.percentile(dist, 100*alpha/2))
            boot_hi.append(np.percentile(dist, 100*(1-alpha/2)))
        wf_df['boot_lower'] = boot_lo
        wf_df['boot_upper'] = boot_hi
        return wf_df

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
        wf_df = forecaster.bootstrap_intervals(wf_df, alpha=0.05, n_samples=BOOTSTRAP_SAMPLES)

        # Metrics for SARIMAX Bootstrapped CIs
        def _cov(a, lo, hi): return np.mean((a >= lo) & (a <= hi))*100
        metrics = {
            'Country': country,
            'RMSE': mean_squared_error(wf_df['actual'], wf_df['forecast'], squared=False),
            'MAPE': np.mean(np.abs((wf_df['actual']-wf_df['forecast'])/wf_df['actual']))*100,
            'Bootstrap_Coverage': _cov(wf_df['actual'], wf_df['boot_lower'], wf_df['boot_upper']),
        }
        logging.info(f"{country} BOOTSTRAP CI: {metrics}")

        # Save walk-forward CIs
        wf_df.to_csv(os.path.join(country_dir, 'walkforward_bootstrap.csv'))

        # === Plotting ===
        plt.figure(figsize=(10,4))
        plt.plot(raw, label='Obs')
        plt.plot(wf_df['forecast'], label='Forecast')
        plt.fill_between(wf_df.index, wf_df['boot_lower'], wf_df['boot_upper'], color='orange', alpha=0.18, label='Bootstrapped CI')
        plt.legend(); plt.tight_layout()
        plt.title(f'{country} SARIMAX+Bootstrapped Backtest CIs')
        plt.savefig(os.path.join(country_dir,'backtest_bootstrap_cis.png'))
        plt.close()

    except Exception as e:
        logging.warning(f"{country}: {e}")

if __name__ == '__main__':
    for c in OECD_CODES:
        run_for_country(c)
