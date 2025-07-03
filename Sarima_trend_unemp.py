"""
SARIMAX + GARCH Evaluation & Forecasting Script
================================================

1.  Loads weekly Google-Trends anxiety → monthly → logit transform.
2.  Loads exogenous covariates: Mobility, Consumer Sentiment, Reddit counts, Exam dummy.
3.  Splits data: train on all but last 12 months, test on last 12.
4.  Fits SARIMAX(1,0,1)x(1,1,0,12)+drift via MLE, then Student-t GARCH(1,1) on residuals.
5.  Forecasts 12-month test period: mean via SARIMAX, variance via GARCH → dynamic 95% CI.
6.  Evaluation: MAE, RMSE, MAPE, CI coverage, residual diagnostics (plot + ACF).
7.  Re-fits on full data and generates a 60-month forecast with GARCH-based CI.
8.  Applies a hard bound [0,100] to all forecasts and CIs and saves CSV & plots.

Dependencies:
    pandas, numpy, matplotlib, requests, statsmodels, scikit-learn, arch

Install:
    pip install pandas numpy matplotlib requests statsmodels scikit-learn arch
"""

import os
import warnings
import io
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from arch import arch_model

# suppress warnings
warnings.filterwarnings('ignore')

# Settings
DATA_PATH       = r"C:\Users\Mega-Pc\Desktop\Timeseries\mental_health_trends.csv"
MOBILITY_CSV    = "google_mobility_work.csv"
REDDIT_CSV      = "reddit_anxiety_posts.csv"
FRED_SERIES     = "UMCSENT"
TEST_MONTHS     = 12
FORECAST_MONTHS = 60
EPS             = 1e-3
Z95             = 1.96  # z-value for 95% CI
OUT_DIR         = r"C:\Users\Mega-Pc\Desktop\Timeseries\sarimax_evaluation"

os.makedirs(OUT_DIR, exist_ok=True)

# ----- Helper Functions -----

def to_logit(x):
    x_clip = x.clip(0.1, 99.9)
    p = (x_clip + EPS) / (100 + 2 * EPS)
    return np.log(p / (1 - p))

def from_logit(z):
    z_arr = np.array(z)
    p = 1 / (1 + np.exp(-z_arr))
    return (100 + 2 * EPS) * p - EPS

def dedup(s):
    return s[~s.index.duplicated()]

# ----- Exogenous Data Loaders -----

def load_mobility():
    url = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
    if not os.path.exists(MOBILITY_CSV):
        pd.read_csv(url, low_memory=False).to_csv(MOBILITY_CSV, index=False)
    df = pd.read_csv(MOBILITY_CSV, low_memory=False)
    df = df.query("sub_region_1.isna()")[['date','workplaces_percent_change_from_baseline']]
    df.columns = ['date','mob_work']
    df['date'] = pd.to_datetime(df['date']) + pd.offsets.MonthBegin()
    df = df.drop_duplicates('date')
    df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
    ser = df.groupby('month')['mob_work'].mean()
    ser.index.name = 'date'
    return ser

def load_sentiment():
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={FRED_SERIES}"
    try:
        r = requests.get(url)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text)).iloc[:, :2]
        df.columns = ['date','cons_sent']
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).drop_duplicates('date')
        df['month'] = df['date'].dt.to_period('M').dt.to_timestamp()
        ser = df.groupby('month')['cons_sent'].mean()
        ser.index.name = 'date'
        return ser
    except Exception:
        return pd.Series(0, name='cons_sent')

def load_reddit():
    try:
        df = pd.read_csv(REDDIT_CSV).iloc[:, :2]
        df.columns = ['date','reddit_cnt']
        df['date'] = pd.to_datetime(df['date']) + pd.offsets.MonthBegin()
        df = df.drop_duplicates('date')
        ser = df.set_index('date')['reddit_cnt'].resample('MS').ffill()
        ser.name = 'reddit_cnt'
        return ser
    except Exception:
        return pd.Series(0, name='reddit_cnt')

# 1) Load & Prepare Target

df_week = pd.read_csv(DATA_PATH, parse_dates=['date']).set_index('date').asfreq('W')
raw = dedup(df_week.resample('MS').mean()['anxiety'].dropna())
target = to_logit(raw)

# 2) Load & Align Exogenous

e_mob  = load_mobility().reindex(raw.index).interpolate('time').fillna(0)
e_sent = load_sentiment().reindex(raw.index).interpolate('time').fillna(0)
e_red  = load_reddit().reindex(raw.index).fillna(0)
exog   = pd.concat([e_mob, e_sent, e_red], axis=1)
exog.columns = ['mob_work','cons_sent','reddit_cnt']
exog['exam_dummy'] = raw.index.month.isin([1,6,12]).astype(int)
for c in exog.columns:
    std = exog[c].std()
    exog[c] = (exog[c] - exog[c].mean()) / (std if std else 1)

# 3) Train-Test Split

y_train   = target.iloc[:-TEST_MONTHS]
y_test    = target.iloc[-TEST_MONTHS:]
raw_train = raw.iloc[:-TEST_MONTHS]
raw_test  = raw.iloc[-TEST_MONTHS:]
X_train   = exog.iloc[:-TEST_MONTHS]
X_test    = exog.iloc[-TEST_MONTHS:]

# 4) Fit SARIMAX

order, seasonal = (1,0,1), (1,1,0,12)
model_sar = SARIMAX(
    y_train, exog=X_train,
    order=order,
    seasonal_order=seasonal,
    trend='t',
    enforce_stationarity=False,
    enforce_invertibility=False
)
res_sar = model_sar.fit(disp=False)

# 5) Fit Student-t GARCH

resid = res_sar.resid.dropna()
res_garch = arch_model(resid, vol='Garch', p=1, q=1, dist='StudentsT').fit(disp=False)

# 6) Forecast Test Period

fc = res_sar.get_forecast(steps=TEST_MONTHS, exog=X_test)
mean_log = fc.predicted_mean
var_test = res_garch.forecast(horizon=TEST_MONTHS).variance.values[-1]
lower_l  = mean_log - Z95 * np.sqrt(var_test)
upper_l  = mean_log + Z95 * np.sqrt(var_test)

forecast = from_logit(mean_log).clip(0,100)
lower_ci = from_logit(lower_l).clip(0,100)
upper_ci = from_logit(upper_l).clip(0,100)

df_test = pd.DataFrame({
    'actual':    raw_test,
    'forecast':  forecast,
    'lower_ci':  lower_ci,
    'upper_ci':  upper_ci
}, index=raw_test.index)

# 7) Evaluation

mae      = mean_absolute_error(df_test.actual, df_test.forecast)
rmse     = mean_squared_error(df_test.actual, df_test.forecast, squared=False)
mape     = (np.abs(df_test.actual - df_test.forecast) / df_test.actual).mean() * 100
coverage = ((df_test.actual >= df_test.lower_ci) & (df_test.actual <= df_test.upper_ci)).mean() * 100
print(f"MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%, Coverage={coverage:.1f}%")

# Residual Diagnostics: plot + ACF
plt.figure(figsize=(8,4))
resids = df_test.forecast - df_test.actual
plt.axhline(0, ls='--', color='gray')
plt.plot(resids.index, resids, marker='o')
plt.title('Residuals (Test)')
plt.savefig(os.path.join(OUT_DIR,'residuals_test.png'))
plt.close()

# Residual ACF plot with safe lag
max_lags = min(len(resids) - 1, 12)
if max_lags > 0:
    plt.figure(figsize=(6,4))
    plot_acf(resids, lags=max_lags)
    plt.title('Residual ACF (Test)')
    plt.savefig(os.path.join(OUT_DIR,'residual_acf.png'))
    plt.close()
else:
    print("Warning: Not enough data points for residual ACF plot")

plt.figure(figsize=(6,4))

#plot_acf(resids, lags=12)
plt.title('Residual ACF (Test)')
plt.savefig(os.path.join(OUT_DIR,'residual_acf.png'))
plt.close()

plt.figure(figsize=(10,5))
plt.plot(raw_train.index, raw_train, 'k-', label='Train')
plt.plot(df_test.index, df_test.actual, 'k--', label='Actual')
plt.plot(df_test.index, df_test.forecast, 'b-', label='Forecast')
plt.fill_between(df_test.index, df_test.lower_ci, df_test.upper_ci, color='b', alpha=0.2)
plt.legend(); plt.title('Test Forecast vs Actual')
plt.savefig(os.path.join(OUT_DIR,'test_forecast_eval.png'))
plt.close()

# 8) Refit Full & 60-month Forecast

res_full = SARIMAX(
    target, exog=exog,
    order=order,
    seasonal_order=seasonal,
    trend='t',
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)
resid_f = res_full.resid.dropna()
res_g_f = arch_model(resid_f, vol='Garch', p=1, q=1, dist='StudentsT').fit(disp=False)

fc_f = res_full.get_forecast(
    steps=FORECAST_MONTHS,
    exog=pd.concat([exog.iloc[-12:]]*5, ignore_index=True)
         .iloc[:FORECAST_MONTHS]
         .set_index(pd.date_range(
             raw.index[-1] + pd.offsets.MonthBegin(),
             periods=FORECAST_MONTHS,
             freq='MS'
         ))
)
mean_log_f = fc_f.predicted_mean
var_full   = res_g_f.forecast(horizon=FORECAST_MONTHS).variance.values[-1]

lower_f_l = mean_log_f - Z95 * np.sqrt(var_full)
upper_f_l = mean_log_f + Z95 * np.sqrt(var_full)
idx_f     = pd.date_range(
    raw.index[-1] + pd.offsets.MonthBegin(),
    periods=FORECAST_MONTHS,
    freq='MS'
)
fcast     = pd.Series(from_logit(mean_log_f).clip(0,100), index=idx_f)
lower_f   = pd.Series(from_logit(lower_f_l).clip(0,100), index=idx_f)
upper_f   = pd.Series(from_logit(upper_f_l).clip(0,100), index=idx_f)

pd.DataFrame({
    'forecast': fcast,
    'lower_ci': lower_f,
    'upper_ci': upper_f
}).to_csv(os.path.join(OUT_DIR,'forecast_60m.csv'), index_label='date')

plt.figure(figsize=(12,5))
plt.plot(raw, 'k-', label='Observed')
plt.plot(fcast, 'b-', label='Forecast')
plt.fill_between(idx_f, lower_f, upper_f, color='b', alpha=0.2)
plt.legend(); plt.title('5-Year Forecast with GARCH CI')
plt.savefig(os.path.join(OUT_DIR,'forecast_5y_eval.png'))

print(f"All artifacts saved to {OUT_DIR}")
