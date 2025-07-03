import pandas as pd
import matplotlib.pyplot as plt
import os

## Set up output directory
output_dir = r"C:\Users\Mega-Pc\Desktop\Timeseries\plots"
os.makedirs(output_dir, exist_ok=True)

## Load and Prepare the Data
path = r"C:\Users\Mega-Pc\Desktop\Timeseries\mental_health_trends.csv"
df = pd.read_csv(path, parse_dates=["date"])
df.set_index("date", inplace=True)
df = df.asfreq('W')  # Ensure consistent weekly frequency

## Time Series Line Plots
plt.figure(figsize=(25, 10))
df.plot(ax=plt.gca(), title="Search Interest Over Time")
plt.ylabel("Search Interest")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "01_timeseries_all_keywords.png"))
plt.show()

## Seasonal Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(df["anxiety"], model='additive', period=52)
fig = result.plot()
fig.suptitle("Seasonal Decomposition of Anxiety", fontsize=16)
fig.set_size_inches(14, 8)
plt.tight_layout()
fig.savefig(os.path.join(output_dir, "02_seasonal_decomposition_anxiety.png"))
plt.show()

## ACF Plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig_acf = plot_acf(df["anxiety"].dropna(), lags=52)
plt.title("ACF - Anxiety")
plt.tight_layout()
fig_acf.figure.savefig(os.path.join(output_dir, "03_acf_anxiety.png"))
plt.show()

## PACF Plot
fig_pacf = plot_pacf(df["anxiety"].dropna(), lags=52)
plt.title("PACF - Anxiety")
plt.tight_layout()
fig_pacf.figure.savefig(os.path.join(output_dir, "04_pacf_anxiety.png"))
plt.show()

## Correlation Heatmap
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Between Keywords")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "05_correlation_heatmap.png"))
plt.show()

## Outlier Detection
from scipy.stats import zscore

z_scores = df.apply(zscore)
outliers = (z_scores.abs() > 3)
print("Outlier counts per keyword:\n", outliers.sum())

## Volatility Analysis
volatility = df.std()
print(volatility.sort_values(ascending=False))

## Rolling Std Dev Plot
df[["anxiety", "depression"]].rolling(window=4).std().plot(title="4-week Rolling Std Dev")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "06_rolling_stddev_anxiety_depression.png"))
plt.show()
