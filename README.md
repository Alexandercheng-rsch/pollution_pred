# 🌍 Air Quality Prediction Pipeline  

The goal of this project was to explore whether **meteorological data alone** can be used to predict **air pollution levels in the UK**.  

I built an end-to-end pipeline that collected data from the **OpenAQ API** (pollution) and the **Meteo API** (weather), processed and combined the datasets, engineered meaningful features, and trained models to predict two pollutants: **Ozone (O₃)** and **Particulate Matter (PM2.5)**.  

To make the results accessible, I also deployed an interactive **Streamlit app** where users can explore predictions and trends.  
![example](https://github.com/Alexandercheng-rsch/pollution_prediction/blob/main/images/example.png)

---

## 🔍 What I Did  

- **Data Collection**  
  - Pulled pollution data from OpenAQ.  
  - Pulled meteorological data from Meteo.  

- **Data Engineering & Feature Engineering**  
  - Cleaned and merged datasets.  
  - Added **rolling averages** and **lag features** to capture temporal dependencies.  
  - Created interaction terms (e.g. **humidity × wind speed**).  
  - Encoded cyclical time using **sin(hour)** and **cos(hour)** to represent daily cycles.  
  - Handled missing values and outliers.  

- **Modeling**  
  - **O₃** → regression model with hyperparameter tuning.  
  - **PM2.5** → a **two-stage pipeline**:  
    1. Classifier predicts whether a measurement is a **peak** or **non-peak**.  
    2. Two separate regressors are used for peak and non-peak cases.  

- **Deployment**  
  - Built a Streamlit app so results can be explored interactively.  

---

## ⚙️ Tech Stack  

- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib/Seaborn, Streamlit)  
- **APIs**: OpenAQ, Meteo  
- **Deployment**: Streamlit Cloud  

---

## 📂 Project Structure  

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
(pollution-prediction.streamlit.app)
## Demo
Click the thumbnail below to watch a demo of the Streamlit app in action:
[![Watch the video](https://img.youtube.com/vi/a4UmjSwL_ds/0.jpg)](https://youtu.be/a4UmjSwL_ds)

