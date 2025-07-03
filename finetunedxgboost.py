import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from xgboost import XGBRegressor
import optuna
from scipy.optimize import minimize_scalar
import pycountry

warnings.filterwarnings("ignore")

# === Logging setup ===
LOG_PATH = r"C:\Users\Mega-Pc\Desktop\Timeseries\blendedoutput\metrics.log"
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

# === Filepaths ===
DATA_PATH = r"C://Users//Mega-Pc//Desktop//Timeseries//mental_health_trends.csv"
MOBILITY_PATH = r"C://Users//Mega-Pc//Desktop//Timeseries//Global_Mobility_Report.csv"
INFLATION_PATH = r"C://Users//Mega-Pc//Desktop//Timeseries//OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,+.M.N.CPI.._T.N.GY+_Z.csv"
UNEMPLOY_PATH = r"C://Users//Mega-Pc//Desktop//Timeseries//OECD.SDD.TPS,DSD_LFS@DF_IALFS_INDIC,+.UNE_LF_M...Y._T.Y_GE15..M.csv"
OUT_DIR = r"C://Users//Mega-Pc//Desktop//Timeseries//blendedoutput"
FOLDS, HORIZON = 4, 12
EPS = 1e-3
FORECAST_MONTHS = 60

os.makedirs(OUT_DIR, exist_ok=True)

OECD_ISO2 = [
    "AU","AT","BE","CA","CL","CO","CZ","CH","DE","DK","EE","ES","FI","FR","GB","GR","HU",
    "IE","IL","IS","IT","JP","KR","LT","LU","LV","MX","NL","NO","NZ","PL","PT","SE","SI","SK","TR","US"
]

def iso2_to_iso3(iso2):
    try:
        return pycountry.countries.get(alpha_2=iso2).alpha_3
    except:
        return None

def get_mobility_iso2(mobility_csv_path):
    df = pd.read_csv(mobility_csv_path, low_memory=False)
    codes = df['country_region_code'].dropna().unique()
    codes = [str(c).upper() for c in codes if isinstance(c, str) and len(c) == 2]
    return set(codes)

def load_oecd_long_format(csv_path):
    for sep in [',', ';']:
        df = pd.read_csv(csv_path, sep=sep, low_memory=False)
        if {'REF_AREA','TIME_PERIOD','OBS_VALUE'}.issubset(df.columns):
            break
    df = df[['REF_AREA', 'TIME_PERIOD', 'OBS_VALUE']].copy()
    df['REF_AREA'] = df['REF_AREA'].astype(str).str.strip().str.upper()
    df['TIME_PERIOD'] = pd.to_datetime(df['TIME_PERIOD'], errors='coerce')
    df['OBS_VALUE'] = (
        df['OBS_VALUE'].astype(str)
        .str.replace(',', '.')
        .str.replace(' ', '')
        .str.extract(r'(\d+\.?\d*)')[0]
        .astype(float)
    )
    df = df.groupby(['TIME_PERIOD', 'REF_AREA'], as_index=False)['OBS_VALUE'].mean()
    wide = df.pivot(index='TIME_PERIOD', columns='REF_AREA', values='OBS_VALUE')
    return wide

def get_available_oecd_countries(mobility_csv_path, infl_wide, unemp_wide, oecd_iso2):
    mob_iso2 = get_mobility_iso2(mobility_csv_path)
    infl_iso3 = set(infl_wide.columns)
    unemp_iso3 = set(unemp_wide.columns)
    oecd_iso3 = set([iso2_to_iso3(c) for c in oecd_iso2 if iso2_to_iso3(c)])
    intersection_iso2 = []
    for iso2 in mob_iso2:
        iso3 = iso2_to_iso3(iso2)
        if (
            iso2 in oecd_iso2 and
            iso3 is not None and
            iso3 in infl_iso3 and
            iso3 in unemp_iso3 and
            iso3 in oecd_iso3
        ):
            intersection_iso2.append(iso2)
    print("Mobility ISO2 (from Google):", sorted(mob_iso2))
    print("Inflation ISO3 (from file):", sorted(infl_iso3))
    print("Unemployment ISO3 (from file):", sorted(unemp_iso3))
    print("OECD ISO3:", sorted(oecd_iso3))
    print("Intersection (final OECD countries ISO2):", sorted(intersection_iso2))
    return sorted(intersection_iso2)

inflation_monthly = load_oecd_long_format(INFLATION_PATH)
unemployment_monthly = load_oecd_long_format(UNEMPLOY_PATH)

OECD_COUNTRIES = get_available_oecd_countries(
    MOBILITY_PATH, inflation_monthly, unemployment_monthly, OECD_ISO2
)
print("OECD_COUNTRIES for main loop:", OECD_COUNTRIES)

def make_holiday_dummy(idx, country=None):
    holiday_months = [1, 6, 7, 8, 12]
    return idx.month.isin(holiday_months).astype(int)

class DataPipeline:
    def __init__(self, mobility_path, country_code):
        self.mobility_path = mobility_path
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

    def load_exog(self):
        df = pd.read_csv(self.mobility_path, low_memory=False)
        country = self.country
        mob = df[df['country_region_code'] == country]
        mob = mob[['date', 'workplaces_percent_change_from_baseline']]
        mob['date'] = pd.to_datetime(mob['date']).dt.to_period('M').dt.to_timestamp()
        mob = mob.groupby('date')['workplaces_percent_change_from_baseline'].mean()
        return mob

    def build_exogenous(self, raw_index):
        exog = pd.DataFrame(index=raw_index)
        mob = self.load_exog()
        exog['mob_work'] = mob.reindex(raw_index)
        iso3 = iso2_to_iso3(self.country)
        exog['inflation'] = inflation_monthly[iso3].reindex(raw_index)
        exog['unemployment'] = unemployment_monthly[iso3].reindex(raw_index)
        exog['holiday_dummy'] = make_holiday_dummy(raw_index, self.country)
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
        z = 1.96
        wf_df['garch_lower'] = DataPipeline.from_logit(wf_df['logit_pred'] - z * sigmas)
        wf_df['garch_upper'] = DataPipeline.from_logit(wf_df['logit_pred'] + z * sigmas)
        return wf_df

    def final_forecast(self, logit, exog, steps=60):
        fut_idx = pd.date_range(logit.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq='MS')
        mob_work_last12 = exog['mob_work'].iloc[-12:].mean()
        inflation_last12 = exog['inflation'].iloc[-12:].mean()
        unemployment_val = exog['unemployment'].iloc[-1]
        holiday_dummy = make_holiday_dummy(fut_idx, None)
        future_exog = pd.DataFrame({
            'mob_work': [mob_work_last12]*steps,
            'inflation': [inflation_last12]*steps,
            'unemployment': [unemployment_val]*steps,
            'holiday_dummy': holiday_dummy,
        }, index=fut_idx)
        model_fit = self.fit_sarimax(logit, exog, self.order, self.seasonal_order)
        fc = model_fit.get_forecast(steps, exog=future_exog)
        fc_mean = DataPipeline.from_logit(fc.predicted_mean)
        fc_ci = fc.conf_int().apply(DataPipeline.from_logit)
        return fc_mean, fc_ci

def tune_xgb(X_train, y_train):
    X_t, X_v, y_t, y_v = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": 42,
        }
        model = XGBRegressor(**params)
        model.fit(
            X_t, y_t,
            eval_set=[(X_v, y_v)],
            verbose=False
        )
        preds = model.predict(X_v)
        return mean_squared_error(y_v, preds)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=40, show_progress_bar=False)
    return study.best_params

def optimize_blend(y_true, sarimax_preds, xgb_preds, metric='rmse'):
    def rmse(blend):
        blend_pred = blend * sarimax_preds + (1-blend) * xgb_preds
        return np.sqrt(np.mean((y_true - blend_pred)**2))
    def mape(blend):
        blend_pred = blend * sarimax_preds + (1-blend) * xgb_preds
        return np.mean(np.abs((y_true - blend_pred) / y_true)) * 100
    objective = rmse if metric == 'rmse' else mape
    res = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    return res.x

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
        pipeline = DataPipeline(MOBILITY_PATH, country)
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

        # === XGBoost Tuning & Training ===
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

        xgb_params = tune_xgb(X_train, y_train)
        xgb = XGBRegressor(**xgb_params)
        xgb.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False
        )
        xgb_preds = xgb.predict(X_test)

        wf_df['xgb_pred'] = xgb_preds

        # --- Blend optimization ---
        blend_ratio = optimize_blend(
            wf_df['actual'].values,
            wf_df['forecast'].values,
            wf_df['xgb_pred'].values,
            metric='rmse'
        )
        wf_df['blend_pred'] = blend_ratio * wf_df['forecast'] + (1-blend_ratio) * wf_df['xgb_pred']

        def _cov(a, lo, hi): return np.mean((a >= lo) & (a <= hi))*100
        metrics = {
            'Country': country,
            'SARIMAX_RMSE': mean_squared_error(wf_df['actual'], wf_df['forecast'], squared=False),
            'SARIMAX_MAPE': np.mean(np.abs((wf_df['actual']-wf_df['forecast'])/wf_df['actual']))*100,
            'SARIMAX_GARCH_Coverage': _cov(wf_df['actual'], wf_df['garch_lower'], wf_df['garch_upper']),
            'XGB_RMSE': mean_squared_error(wf_df['actual'], wf_df['xgb_pred'], squared=False),
            'XGB_MAPE': np.mean(np.abs((wf_df['actual']-wf_df['xgb_pred'])/wf_df['actual']))*100,
            'BLEND_RMSE': mean_squared_error(wf_df['actual'], wf_df['blend_pred'], squared=False),
            'BLEND_MAPE': np.mean(np.abs((wf_df['actual']-wf_df['blend_pred'])/wf_df['actual']))*100,
            'Blend_Ratio': blend_ratio
        }
        logging.info(f"{country} BLEND metrics: {metrics}")

        wf_df.to_csv(os.path.join(country_dir, 'walkforward_garch_xgb_blend.csv'))

        # === Plotting (backtest) ===
        plt.figure(figsize=(10,4))
        plt.plot(raw, label='Obs')
        plt.plot(wf_df['forecast'], label='SARIMAX Forecast')
        plt.plot(wf_df['xgb_pred'], label='XGBoost')
        plt.plot(wf_df['blend_pred'], label='Blend')
        plt.fill_between(wf_df.index, wf_df['garch_lower'], wf_df['garch_upper'], color='purple', alpha=0.18, label='GARCH CI')
        plt.legend(); plt.tight_layout()
        plt.title(f'{country} SARIMAX+GARCH & XGBoost Blend Backtest')
        plt.savefig(os.path.join(country_dir,'backtest_garch_xgb_blend.png'))
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
    for c in OECD_COUNTRIES:
        run_for_country(c)
