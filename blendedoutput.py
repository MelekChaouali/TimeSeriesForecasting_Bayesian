import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime

# === SETTINGS ===
OUT_DIR = 'blendedoutput'  # Relative path for GitHub and deployment
TITLE = "OECD Mental Health & Economic Indicators Dashboard"
DESCRIPTION = """
Explore the backtested forecasts and performance metrics for anxiety trends across OECD countries.  
Compare SARIMAX-GARCH, XGBoost, and their blend, with confidence intervals and key metrics.
"""

def get_available_countries(out_dir):
    if not os.path.exists(out_dir):
        return []
    return sorted([d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))])

def get_latest_update(country_dir):
    try:
        files = [os.path.join(country_dir, f) for f in os.listdir(country_dir)]
        latest_file = max(files, key=os.path.getmtime)
        update_date = datetime.fromtimestamp(os.path.getmtime(latest_file))
        return update_date.strftime("%d/%m/%Y %H:%M")  # Day/Month/Year Hour:Minute
    except:
        return "Unknown"

def display_metrics(wf_df):
    actual = wf_df['actual']
    sarimax = wf_df['forecast']
    xgb = wf_df['xgb_pred']
    blend = wf_df['blend_pred']

    sarimax_rmse = mean_squared_error(actual, sarimax, squared=False)
    sarimax_mape = np.mean(np.abs((actual - sarimax) / actual)) * 100
    xgb_rmse = mean_squared_error(actual, xgb, squared=False)
    xgb_mape = np.mean(np.abs((actual - xgb) / actual)) * 100
    blend_rmse = mean_squared_error(actual, blend, squared=False)
    blend_mape = np.mean(np.abs((actual - blend) / actual)) * 100
    blend_ratio = np.nanmean((blend - xgb) / (sarimax - xgb + 1e-8))  # fallback

    col1, col2, col3 = st.columns(3)
    col1.metric("SARIMAX RMSE", f"{sarimax_rmse:.2f}")
    col1.metric("SARIMAX MAPE", f"{sarimax_mape:.2f}%")
    col2.metric("XGBoost RMSE", f"{xgb_rmse:.2f}")
    col2.metric("XGBoost MAPE", f"{xgb_mape:.2f}%")
    col3.metric("Blend RMSE", f"{blend_rmse:.2f}")
    col3.metric("Blend MAPE", f"{blend_mape:.2f}%")
    col3.metric("Blend Ratio", f"{blend_ratio:.2f}")

def plot_backtest(wf_df, save_path, country):
    dates = wf_df.index
    obs = wf_df['actual']
    sarimax = wf_df['forecast']
    xgb = wf_df['xgb_pred']
    blend = wf_df['blend_pred']

    ci = False
    if 'garch_lower' in wf_df and 'garch_upper' in wf_df:
        lower_ci = wf_df['garch_lower']
        upper_ci = wf_df['garch_upper']
        ci = True

    fig, ax = plt.subplots(figsize=(8, 4))  # Smaller figure size
    ax.plot(dates, obs, label='Obs')
    ax.plot(dates, sarimax, label='SARIMAX Forecast')
    ax.plot(dates, xgb, label='XGBoost')
    ax.plot(dates, blend, label='Blend')
    if ci:
        ax.fill_between(dates, lower_ci, upper_ci, color='purple', alpha=0.3, label='GARCH CI')
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title(f'{country} SARIMAX+GARCH & XGBoost Blend Backtest', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.3)
    plt.close()

# === STREAMLIT DASHBOARD ===
st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)
st.markdown(DESCRIPTION)

# --- Country selection ---
countries = get_available_countries(OUT_DIR)
if not countries:
    st.error("No country output found in output directory.")
    st.stop()

country = st.selectbox("Select Country", countries)
country_dir = os.path.join(OUT_DIR, country)

# --- Load walkforward and show metrics ---
wf_path = os.path.join(country_dir, 'walkforward_garch_xgb_blend.csv')
backtest_img = os.path.join(country_dir, 'backtest_garch_xgb_blend.png')
forecast_img = os.path.join(country_dir, 'forecast_5year.png')

if os.path.exists(wf_path):
    wf_df = pd.read_csv(wf_path, index_col=0, parse_dates=True)
    st.subheader("Forecast Performance Metrics")
    display_metrics(wf_df)

    # Regenerate the backtest plot (optional: avoid re-plotting if not needed)
    plot_backtest(wf_df, backtest_img, country)
else:
    st.warning("Metrics not found for this country.")
    wf_df = None

# --- Show plots side-by-side ---
st.subheader("Forecast Visualizations")

col1, col2 = st.columns(2)

with col1:
    if os.path.exists(backtest_img):
        st.image(backtest_img, caption="Backtest (SARIMAX+GARCH, XGBoost, Blend)", use_container_width=True)
    else:
        st.warning("Backtest plot not found for this country.")

with col2:
    if os.path.exists(forecast_img):
        st.image(forecast_img, caption="5-year SARIMAX Forecast with 95% CI", use_container_width=True)
    else:
        st.warning("5-year forecast plot not found for this country.")

# --- Show underlying dataframes (optional, toggle) ---
with st.expander("Show forecast and actual data table"):
    if wf_df is not None:
        st.dataframe(wf_df, use_container_width=True)
    else:
        st.info("No walkforward data to show.")

# --- Footer ---
st.markdown("---")
st.caption(
    f"Latest data update: {get_latest_update(country_dir)}  \n"
    "Contact us for issues or improvements: "
    "[mohamed-aziz.ben-ammar@etu.unistra.fr](mailto:mohamed-aziz.ben-ammar@etu.unistra.fr)"
)
