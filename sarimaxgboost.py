import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from xgboost import XGBRegressor
import optuna

# === Logging setup ===
LOG_PATH = r"C:\Users\Mega-Pc\Desktop\Timeseries\sarimax_garchxgboost_output\metrics.log"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filename=LOG_PATH,
    filemode='w',
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
OUT_DIR = os.path.join(MOBILITY_DIR, "sarimax_garchxgboost_output")
INFLATION_PATH = r"C:\Users\Mega-Pc\Desktop\Timeseries\export-2025-06-26T09_14_43.261Z.csv"
UNEMPLOY_PATH = r"C:\Users\Mega-Pc\Desktop\Timeseries\export-2025-06-26T08_55_28.470Z.csv"
FOLDS, HORIZON = 4, 12
EPS = 1e-3
JOBS = 1
FORECAST_MONTHS = 60

os.makedirs(OUT_DIR, exist_ok=True)

# --- Inflation Loader ---
def load_inflation_monthly(path):
    # Try to read the inflation CSV with semicolon delimiter and skip the first row if needed
    df = pd.read_csv(path, sep=';', skiprows=2)
    # Check for column names like ['Category', 'All items', ...]
    if not any("All items" in str(c) for c in df.columns):
        print("Columns detected:", df.columns)
        raise KeyError("No 'All items' inflation column found in inflation CSV!")
    # Rename columns if necessary
    df.columns = [c.strip() for c in df.columns]
    # Melt if wide, or use as is if long
    if 'Category' in df.columns:
        # This is a wide table, need to melt
        df = df.melt(id_vars=['Category'], var_name='date', value_name='inflation')
        df = df[df['Category'] == 'All items']
        df = df[['date', 'inflation']]
    elif 'All items' in df.columns:
        # Already tidy
        df = df.rename(columns={'All items': 'inflation', df.columns[0]: 'date'})
        df = df[['date', 'inflation']]
    # Parse dates (should be like 'Mar-90')
    df['date'] = pd.to_datetime(df['date'].str.replace('"', ''), format='%b-%y', errors='coerce')
    df = df.dropna(subset=['date', 'inflation'])
    df['inflation'] = df['inflation'].astype(str).str.replace(',', '.').astype(float)
    return df.set_index('date')['inflation']


inflation_monthly = load_inflation_monthly(INFLATION_PATH)

# --- Unemployment Loader ---
def load_unemployment_map(path):
    df = pd.read_csv(path, sep=';', header=None, skiprows=2, dtype=str)
    df = df.iloc[:, :2]
    df.columns = ['country', 'march_2025']
    df = df.dropna()
    df = df[~df['march_2025'].str.contains(r'[A-Za-z]')]
    df['march_2025'] = df['march_2025'].str.replace(',', '.').astype(float)
    df['country'] = df['country'].str.lower().str.strip()
    return dict(zip(df['country'], df['march_2025']))

unemployment_map = load_unemployment_map(UNEMPLOY_PATH)

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

country_name_map = {
    'AT': 'austria', 'AU': 'australia', 'BE': 'belgium', 'CA': 'canada', 'CL': 'chile', 'CZ': 'czechia',
    'DK': 'denmark', 'EE': 'estonia', 'FI': 'finland', 'FR': 'france',  # Add more if you add more countries
}

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
        exog = pd.DataFrame(index=raw_index)
        mob = self.load_exog(self.mobility_csv, 'mob_work')
        exog['mob_work'] = mob.reindex(raw_index)
        exog['inflation'] = inflation_monthly.reindex(raw_index)
        c_name = country_name_map.get(self.country, self.country).lower()
        unem_val = unemployment_map.get(c_name, np.nan)
        exog['unemployment'] = unem_val
        exog['exam_dummy'] = raw_index.month.isin([1,6,12]).astype(int)
        for c in exog.columns:
            std = exog[c].std()
            exog[c] = (exog[c] - exog[c].mean())/(std if std>0 else 1)
        exog = exog.interpolate().fillna(0)
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
            except:
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
        forecast = garch_res.forecast(start=0, horizon=1)
        sigmas = forecast.variance.values.flatten()[:len(wf_df)]
        sigmas = np.sqrt(sigmas)
        z = 1.96  # 95%
        wf_df['garch_lower'] = DataPipeline.from_logit(wf_df['logit_pred'] - z * sigmas)
        wf_df['garch_upper'] = DataPipeline.from_logit(wf_df['logit_pred'] + z * sigmas)
        return wf_df

    def final_forecast(self, logit, exog, steps=60):
        fut_idx = pd.date_range(logit.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq='MS')
        mob_work_last12 = exog['mob_work'].iloc[-12:].mean()
        inflation_last12 = exog['inflation'].iloc[-12:].mean()
        unemployment_val = exog['unemployment'].iloc[-1]
        exam_dummy = fut_idx.month.isin([1,6,12]).astype(int)
        future_exog = pd.DataFrame({
            'mob_work': [mob_work_last12]*steps,
            'inflation': [inflation_last12]*steps,
            'unemployment': [unemployment_val]*steps,
            'exam_dummy': exam_dummy,
        }, index=fut_idx)
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

        # === SARIMAX+GARCH ===
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda t: objective(t, logit, exog), timeout=180, n_trials=40)
        bp = study.best_params
        order = (bp['p'],bp['d'],bp['q'])
        seasonal = (bp['P'],bp['D'],bp['Q'],12)

        forecaster = ForecastPipeline(order, seasonal)
        wf_df = forecaster.walk_forward(logit, exog)
        wf_df = forecaster.garch_intervals(wf_df)

        # === XGBoost Regressor: train on SARIMAX features + exog ===
        y_full = raw
        sarimax_fit = forecaster.fit_sarimax(logit[:-HORIZON], exog[:-HORIZON], order, seasonal)
        if sarimax_fit is not None:
            fitted_vals = DataPipeline.from_logit(sarimax_fit.fittedvalues)
        else:
            fitted_vals = pd.Series(np.nan, index=raw.index[:-HORIZON])
        df_xgb = exog.copy()
        df_xgb['sarimax_pred'] = np.concatenate([fitted_vals.values, [np.nan]*HORIZON])
        df_xgb = df_xgb.iloc[:-HORIZON]
        X_train = df_xgb.values
        y_train = y_full.iloc[:len(X_train)].values
        X_test = exog.iloc[-HORIZON:].copy()
        X_test['sarimax_pred'] = wf_df['forecast'].values
        xgb = XGBRegressor(n_estimators=100, max_depth=2, learning_rate=0.1, random_state=42)
        xgb.fit(X_train, y_train)
        xgb_preds = xgb.predict(X_test)

        def _cov(a, lo, hi): return np.mean((a >= lo) & (a <= hi))*100
        metrics = {
            'Country': country,
            'SARIMAX_RMSE': mean_squared_error(wf_df['actual'], wf_df['forecast'], squared=False),
            'SARIMAX_MAPE': np.mean(np.abs((wf_df['actual']-wf_df['forecast'])/wf_df['actual']))*100,
            'SARIMAX_GARCH_Coverage': _cov(wf_df['actual'], wf_df['garch_lower'], wf_df['garch_upper']),
            'XGB_RMSE': mean_squared_error(wf_df['actual'], xgb_preds, squared=False),
            'XGB_MAPE': np.mean(np.abs((wf_df['actual']-xgb_preds)/wf_df['actual']))*100
        }
        logging.info(f"{country} SARIMAX+GARCH+XGBoost metrics: {metrics}")

        # Save walk-forward CIs & preds
        wf_df['xgb_pred'] = xgb_preds
        wf_df.to_csv(os.path.join(country_dir, 'walkforward_garch_xgb.csv'))

        # === Plotting (backtest) ===
        plt.figure(figsize=(10,4))
        plt.plot(raw, label='Obs')
        plt.plot(wf_df['forecast'], label='SARIMAX Forecast')
        plt.plot(wf_df['xgb_pred'], label='XGBoost')
        plt.fill_between(wf_df.index, wf_df['garch_lower'], wf_df['garch_upper'], color='purple', alpha=0.18, label='GARCH CI')
        plt.legend(); plt.tight_layout()
        plt.title(f'{country} SARIMAX+GARCH & XGBoost Backtest')
        plt.savefig(os.path.join(country_dir,'backtest_garch_xgb.png'))
        plt.close()

        # === Final Forecast (SARIMAX 5y out-of-sample) ===
        fc_mean, fc_ci = forecaster.final_forecast(logit, exog, steps=FORECAST_MONTHS)
        fc_mean.to_csv(os.path.join(country_dir, "forecast_5year.csv"))
        fc_ci.to_csv(os.path.join(country_dir, "forecast_5year_ci.csv"))
        plt.figure(figsize=(12,4))
        plt.plot(raw, label='Observed')
        plt.plot(fc_mean, label='5yr SARIMAX Forecast')
        plt.fill_between(fc_ci.index, fc_ci.iloc[:,0], fc_ci.iloc[:,1], alpha=0.2, label='95% CI')
        plt.legend()
        plt.title(f"{country} 5-year SARIMAX Forecast")
        plt.tight_layout()
        plt.savefig(os.path.join(country_dir, 'forecast_5year.png'))
        plt.close()

    except Exception as e:
        logging.warning(f"{country}: {e}")

if __name__ == '__main__':
    for c in OECD_CODES:
        run_for_country(c)
