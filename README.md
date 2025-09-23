# Air Quality Prediction Pipeline

![example](https://github.com/Alexandercheng-rsch/pollution_prediction/blob/main/images/example.png)

## 🎯 Motivation

Air pollution is one of the most pressing environmental health challenges facing the UK today, contributing to thousands of premature deaths annually and affecting millions with respiratory conditions. While traditional air quality monitoring relies on expensive sensor networks with limited coverage, **what if we could predict pollution levels using readily available weather data?**

This project explores a fundamental question: **Can meteorological data alone accurately predict air pollution levels?** By leveraging the strong relationship between weather patterns and pollutant dispersion, I aimed to create an accessible prediction system that could:

- **Complement existing monitoring networks** with predictions for areas without sensors
- **Provide early warnings** for high pollution episodes
- **Support public health decisions** with reliable air quality forecasts
- **Demonstrate the power of data science** in addressing real-world environmental challenges

---

## 🚀 What I Built

I developed an end-to-end machine learning pipeline that transforms weather data into air quality predictions for two critical pollutants: **Ozone (O₃)** and **Particulate Matter (PM2.5)**.

### Key Components:

- **Data Collection Pipeline**
  - Automated data retrieval from OpenAQ API (pollution measurements)
  - Weather data integration from Meteo API
  - Robust handling of API rate limits and large dataset collection

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

## 📊 Key Results

### Model Performance & Key Question: Can Meteorological Data Predict Pollution?

**Yes, meteorological data can effectively predict air pollution levels**, but with important nuances:

- **Strong Performance for Ambient Pollution**: Models perform well predicting typical background pollution levels
- **Struggles with Peak Episodes**: Both O₃ and PM2.5 models face challenges accurately predicting sudden pollution spikes
- **Two-stage PM2.5 approach** helped address peak prediction challenges with specialized models

**Strongest Meteorological Predictors:**
- **For O₃**: Mean sea level pressure, temperature, surface pressure, and wind direction
- **For PM2.5**: Mean sea level pressure, surface pressure, monitoring station location, and temperature

### Meteorological Patterns During Pollution Peaks

**Ozone Peak Conditions:**
- **High temperatures** (30-32°C mid-afternoon) accelerating photochemical ozone formation
- **Low humidity** during daytime hours - dry conditions favor ozone chemistry
- **Light winds** throughout the day - minimal pollutant dispersion
- **High, stable pressure systems** - creating clear skies and stagnant atmospheric conditions
- **Classic anticyclonic pattern**: High pressure + light winds + strong solar radiation = ideal ozone formation

**Temperature-Humidity Inverse Relationship:**
- High humidity overnight when temperatures are low
- Humidity drops dramatically during peak temperature/ozone hours
- This pattern is critical for understanding when ozone formation accelerates

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

