import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from datetime import datetime

# === SETTINGS ===
OUT_DIR = "blendedoutput"  # Change this if you need an absolute path!
TITLE = "OECD Mental Health & Economic Indicators Dashboard"
DESCRIPTION = """
Explore the backtested forecasts and performance metrics for anxiety trends across OECD countries.  
Compare SARIMAX-GARCH, XGBoost, and their blend, with confidence intervals and key metrics.  
Now includes Bayesian credible intervals for more robust uncertainty quantification!
"""

EXPLAINER = """
**How to interpret the charts and intervals:**

- **Confidence Intervals (CIs):** Shaded areas around predictions. They give a range in which we expect the real values to fall. The narrower the interval, the more "certain" the model is. 
- **Coverage:** The percentage of real points that actually fall within the interval. Ideally, a 95% CI should contain the actual value 95% of the time.
- **Why Bayesian Intervals?**  
  Traditional intervals (like from GARCH) only look at short-term volatility. Bayesian intervals combine data and prior model knowledge, so they adapt better to uncertainty, structural shifts, and changing trends.  
  **For you:** This means a more honest reflection of what’s predictable—and what’s not.

**In short:**  
- If the actual curve is inside the shaded Bayesian band, the model's prediction is well-calibrated.
- Wider bands = more uncertainty.  
- Bayesian intervals may be wider or narrower, but their "coverage" is more reliable and interpretable.
"""

# --- Get available countries from output directory ---
def get_available_countries(out_dir):
    if not os.path.exists(out_dir):
        st.warning(f"DEBUG: Directory '{out_dir}' not found.")
        return []
    countries = [d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))]
    if not countries:
        st.warning(f"DEBUG: No subfolders found in '{out_dir}'.")
    return sorted(countries)

def get_latest_update(country_dir):
    try:
        files = [os.path.join(country_dir, f) for f in os.listdir(country_dir)]
        latest_file = max(files, key=os.path.getmtime)
        update_date = datetime.fromtimestamp(os.path.getmtime(latest_file))
        return update_date.strftime("%d/%m/%Y %H:%M")  # Day/Month/Year Hour:Minute
    except Exception as e:
        return "Unknown"

def display_metrics(wf_df):
    actual = wf_df['actual']
    sarimax = wf_df['forecast']
    xgb = wf_df['xgb_pred']
    blend = wf_df['blend_pred']

    sarimax_rmse = np.sqrt(mean_squared_error(actual, sarimax))
    sarimax_mape = np.mean(np.abs((actual - sarimax) / actual)) * 100
    xgb_rmse = np.sqrt(mean_squared_error(actual, xgb))
    xgb_mape = np.mean(np.abs((actual - xgb) / actual)) * 100
    blend_rmse = np.sqrt(mean_squared_error(actual, blend))
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

# === STREAMLIT DASHBOARD ===
st.set_page_config(page_title=TITLE, layout="wide")
st.title(TITLE)
st.markdown(DESCRIPTION)
st.info(EXPLAINER)

# --- Country selection ---
countries = get_available_countries(OUT_DIR)
if not countries:
    st.error("No country output found in output directory. Make sure the 'blendedoutput' folder exists and is correctly populated.")
    st.stop()

country = st.selectbox("Select Country", countries)
if not country:
    st.warning("No country selected. Please choose from the dropdown.")
    st.stop()

country_dir = os.path.join(OUT_DIR, country)

# --- Load walkforward and show metrics ---
wf_path = os.path.join(country_dir, 'walkforward_garch_xgb_blend.csv')
if os.path.exists(wf_path):
    wf_df = pd.read_csv(wf_path, index_col=0, parse_dates=True)
    st.subheader("Forecast Performance Metrics")
    display_metrics(wf_df)
else:
    st.warning("Metrics not found for this country.")
    wf_df = None

# --- Show plots side-by-side ---
st.subheader("Forecast Visualizations")

backtest_img = os.path.join(country_dir, 'backtest_garch_xgb_blend.png')
forecast_img = os.path.join(country_dir, 'forecast_5year.png')
bayesian_img = os.path.join(country_dir, f'bayesian_cis_{country}.png')

col1, col2, col3 = st.columns(3)

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

with col3:
    if os.path.exists(bayesian_img):
        st.image(bayesian_img, caption="Bayesian 95% Credible Intervals", use_container_width=True)
    else:
        st.info("Bayesian CI plot not found for this country. Run the Bayesian script to generate.")

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
