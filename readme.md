# Forecasting Mental Health Trends using Mobility, Inflation, and Unemployment Indicators: A Hybrid Bayesian and Machine Learning Approach

## Project Overview

This project provides a forecasting framework to predict mental health trends, specifically anxiety levels, across OECD countries using a hybrid machine learning and Bayesian modeling approach. It integrates macroeconomic indicators (mobility, inflation, unemployment) with advanced time series models (SARIMAX, GARCH), XGBoost regressors, and Bayesian inference for improved uncertainty quantification.

## Project Objective

* Accurately forecast anxiety trends using macroeconomic indicators.
* Integrate hybrid modeling approaches to enhance prediction performance.
* Provide robust uncertainty estimates through Bayesian credible intervals.
* Deliver an interactive dashboard for easy visualization and interpretation.

## Data Sources

* **Mental Health Trends**: Weekly anxiety scores (`mental_health_trends.csv`).
* **Mobility Data**: Google COVID-19 Community Mobility Reports.
* **Economic Data**: OECD Consumer Price Index and Labor Force Statistics.

## Methodology

### 1. Data Preprocessing

* Data alignment and frequency harmonization.
* Logit transformations for anxiety scores.
* Standardization and preparation of exogenous variables.

### 2. Modeling Approaches

* **SARIMAX + GARCH:** Time series forecasting with volatility modeling.
* **XGBoost Regressor:** Enhanced with mobility, inflation, unemployment data, and SARIMAX fitted values.
* **Model Blending:** Optimized combination of SARIMAX and XGBoost using RMSE.

### 3. Bayesian Inference

* Implemented Bayesian regression models using PyMC:

  * **Prior on Mean:** Bayesian regression leveraging blended forecasts as priors.
  * **Prior on Regression Coefficient:** Strong prior belief integrated into regression coefficient (Beta).
* Credible intervals calculated to improve uncertainty estimation.

## Evaluation Metrics

* **RMSE (Root Mean Squared Error)**
* **MAPE (Mean Absolute Percentage Error)**
* **Credible Interval Coverage**

## Dashboard Features

* Developed with **Streamlit** for interactivity.
* Interactive country selection.
* Comparative visualization of SARIMAX, XGBoost, blended forecasts, and Bayesian credible intervals.
* Performance metrics displayed clearly for stakeholders.
* User-friendly explanations for non-technical audiences regarding the significance and interpretation of Bayesian credible intervals.

## Repository Structure

```
├── AnxietyTimeseriesDashboard
│   ├── blendedoutput
│   ├── Dashboard.py
│   ├── bayesian_intervals.py
│   ├── ...
├── exploratory_data_analysis
│   ├── visualizations
│   ├── exploratory_scripts
│   ├── ...
├── OECD data files
├── mental_health_trends.csv
├── README.md
```

## How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit dashboard:

```bash
streamlit run AnxietyTimeseriesDashboard/Dashboard.py
```

## Future Enhancements

* Expand Bayesian modeling to include hierarchical structures.
* Further optimization of model blending using advanced machine learning techniques.
* Real-time data integration for live forecasting updates.

## Contributions and Acknowledgments

**Authors:**
Mohamed-Aziz BEN AMMAR – [mohamed-aziz.ben-ammar@etu.unistra.fr](mailto:mohamed-aziz.ben-ammar@etu.unistra.fr)
Melek Chaouali – [melekch777@gmail.com](mailto:melekch777@gmail.com)

**Contributions:**
Feel free to open an issue or submit pull requests to contribute to this project.

## References

* [Google COVID-19 Community Mobility Reports](https://www.google.com/covid19/mobility/)
* [OECD Data](https://data.oecd.org/)
* XGBoost: Chen & Guestrin, 2016
* SARIMAX & GARCH literature: Box, Jenkins, and Reinsel
* Bayesian inference: Gelman et al., "Bayesian Data Analysis"

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
