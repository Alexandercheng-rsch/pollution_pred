![example](https://github.com/Alexandercheng-rsch/pollution_prediction/blob/main/images/example.png)

## Motivation

Air pollution is one of the most pressing environmental health challenges facing the UK today, contributing to thousands of premature deaths annually and affecting millions with respiratory conditions. While traditional air quality monitoring relies on expensive sensor networks with limited coverage, **what if we could predict pollution levels using readily available weather data?**

This project explores a fundamental question: **Can meteorological data accurately predict air pollution levels?** By analyzing the relationship between weather patterns and pollutant concentrations, I aimed to create a prediction system that could demonstrate:

- **The feasibility of weather-based pollution prediction** using historical data
- **Feature engineering techniques** that capture temporal pollution patterns  
- **Model performance** for different types of pollutants and conditions
- **Insights into meteorological drivers** of air quality variations
- **The power of data science** in addressing environmental challenges

---

## What I Built

I developed an end-to-end machine learning pipeline that transforms weather data into air quality predictions for two critical pollutants: **Ozone (O₃)** and **Particulate Matter (PM2.5)**.

### Key Components:

- **Data Collection Scripts**
  - Scripts to collect historical pollution data from OpenAQ API
  - Scripts to collect historical meteorological data from Meteo API
  - Robust handling of API rate limits for large dataset collection

- **Advanced Feature Engineering**
  - Rolling averages and lag features to capture temporal patterns
  - Interaction terms (e.g., humidity × wind speed) for complex relationships
  - Cyclical time encoding using trigonometric functions
  - Comprehensive data cleaning and outlier detection

- **Intelligent Modeling Approach**
  - **O₃ Prediction**: Optimized regression model with hyperparameter tuning
  - **PM2.5 Prediction**: Novel two-stage pipeline:
    1. Classification model identifies pollution peaks vs. normal conditions
    2. Specialized regressors handle each case with tailored approaches

- **Interactive Web Application**
  - Streamlit-powered dashboard for real-time exploration
  - Visualizations of trends, predictions, and model insights
  - User-friendly interface for non-technical stakeholders

---
## Key Results

### Can Meteorological Data Predict Pollution?

**Yes, meteorological data can effectively predict air pollution levels**, with important nuances:

#### O₃ Prediction Performance

| Model                   | MAE    | R²      | MSE      | RMSE   |
|-------------------------|--------|---------|----------|--------|
| **Custom Regression Model** | 12.49  | 0.46    | 256.94  | 16.03  |
| **Persistence Baseline**    | 19.43  | -0.37   | 651.82  | 25.53  |

**Insights:**
- The custom model significantly outperforms the persistence baseline.  
- Captures typical ozone levels reasonably well, but sudden spikes remain challenging.  
- Demonstrates that meteorological predictors like temperature, pressure, and wind can inform ozone levels.

#### PM2.5 Prediction Performance

| Model                       | MAE   | R²    | MSE    | RMSE  |
|------------------------------|-------|-------|--------|-------|
| **Custom Two-Stage Model**   | 2.07  | 0.58  | 14.73  | 3.84  |
| **Persistence Baseline**     | 2.41  | 0.43  | 20.04  | 4.48  |

**Insights:**
- The custom model outperforms the persistence baseline, especially in capturing temporal patterns and peak events.  
- Overall predictive accuracy is reasonable, though extreme pollution spikes remain challenging.  
- Confirms that meteorological data can serve as a proxy for air quality monitoring, particularly for typical background pollution levels.

#### Strongest Meteorological Predictors
- **For O₃**: Mean sea level pressure, station, temperature, surface pressure, wind direction, lags
- **For PM2.5**: Mean sea level pressure, station, surface pressure, monitoring station, temperature, lags
**Insights:**
- The model can remember characteristics of each station which in turn can improve performance.
- Lags remain an essential insight for the model as it's an indication of the current trend.
### Meteorological Patterns During Pollution Peaks

**Ozone Peak Conditions:**
- **High temperatures** (30–32°C mid-afternoon) accelerate photochemical ozone formation  
- **Low humidity** during daytime hours favors ozone chemistry  
- **Light winds** reduce pollutant dispersion  
- **High, stable pressure systems** create clear skies and stagnant conditions  
- **Classic anticyclonic pattern**: High pressure + light winds + strong solar radiation = ideal ozone formation  

**Temperature-Humidity Inverse Relationship:**
- High humidity overnight when temperatures are low  
- Humidity drops dramatically during peak temperature/ozone hours  
- Critical for understanding when ozone formation accelerates

---

## Frameworks 

- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib/Seaborn, Streamlit)  
- **APIs**: OpenAQ, Meteo  
- **Deployment**: Streamlit Cloud  

---

## Project Structure  

```bash
air-quality-pipeline/
├── src/
│   ├── notebooks/                 # Jupyter notebooks for EDA and preprocessing
│   ├── scripts/                   # API scraping, prediction, Streamlit app
│   ├── ect/                       # Pickled objects (e.g. coords, dates)
│   └── icon/                      # Icons and static assets
├── requirements.txt
└── .gitignore
```
## Live App
Check out my Streamlit app [here](https://pollution-prediction.streamlit.app/).

## Demo
Click the thumbnail below to watch a demo of the Streamlit app in action:
[![Watch the video](https://img.youtube.com/vi/a4UmjSwL_ds/0.jpg)](https://youtu.be/a4UmjSwL_ds)

## Future Enhancements

- Integration of satellite data for improved spatial coverage
- Real-time prediction updates with live weather feeds
- Mobile app development for on-the-go air quality alerts
- Expansion to additional pollutants (NO₂, SO₂, CO)

---

## Why This Matters

This project demonstrates that **accessible meteorological data can serve as a powerful proxy for expensive air quality monitoring systems**. The implications extend beyond technical achievement—this approach could democratize air quality information, particularly benefiting underserved communities lacking comprehensive monitoring infrastructure.

By making air pollution prediction more accessible and cost-effective, we can better protect public health and inform environmental policy decisions.