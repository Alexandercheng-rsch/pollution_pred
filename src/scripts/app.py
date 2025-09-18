import streamlit as st
import pickle
import datetime
import numpy as np 
import time
import pydeck as pdk
import matplotlib.pyplot as plt 
import altair as alt
import random
import xgboost as xgb
import pandas as pd
from matplotlib.colors import rgb2hex
from functions import cmap_continuous, status_color
import matplotlib.cm as cm
import base64
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import os
import gdown
import warnings

warnings.filterwarnings("ignore")



url = "1O9ct9Upf5fLT3sztkAGTzMZJbyWsyy4s"
output = "/tmp/pollution_data"

# Make sure the folder exists
os.makedirs(output, exist_ok=True)

gdown.download_folder(url, output=output, quiet=False)

# -- Cooldown for prediction button
cooldown = 5
# -- Loading valid dates and test data
@st.cache_data
def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

# -- Load arrows for wind dir
@st.cache_data
def load_icon(path):
    with open(path, 'rb') as f:
        file = f.read()
    return 'data:image/png;base64,' + base64.b64encode(file).decode()

green_arrow = load_icon('src/icon/arrow.png')
# -- Set page config
apptitle = 'GW Quickview'

# -- Load XG Models
# -- o3 Model
@st.cache_data
def load_model_o3():
    with open('/tmp/src/models/xg_reg_o3.model', 'rb') as f:
        model = pickle.load(f)
    return model 

# -- PM2.5 Classifier Model
@st.cache_data
def load_model_pm25_classifier():
    model = xgb.XGBClassifier()
    model.load_model('/tmp/src/models/binary_classifier.model')
    return model 

# -- PM2.5 Middle Model
@st.cache_data
def load_model_pm25_middle():
    model = xgb.XGBRegressor()
    model.load_model('/tmp/src/models/xg_mid.model')
    return model 
# -- PM2.5 Upper Model 
@st.cache_data
def load_model_pm25_upper():
    model = xgb.XGBRegressor()
    model.load_model('/tmp/src/models/xg_90.model')
    return model 

# -- Initiate Models
o3_model = load_model_o3()
pm25_classifier_model = load_model_pm25_classifier()
pm25_middle_model = load_model_pm25_middle()
pm25_upper_model = load_model_pm25_upper()

o3_model.set_params(tree_method='hist', device='cpu')
pm25_classifier_model.set_params(tree_method='hist', device='cpu')
pm25_middle_model.set_params(tree_method='hist', device='cpu')
pm25_upper_model.set_params(tree_method='hist', device='cpu')

# -- Load Target Encoders
# -- o3 Model
@st.cache_data
def load_encoder_o3():
    with open('/tmp/src/models/encoder_o3.model', 'rb') as f:
        model = pickle.load(f)
    return model

# -- PM2.5 Model
@st.cache_data
def load_encoder_pm25():
    with open('/tmp/src/models/encoder_pm25.model', 'rb') as f:
        model = pickle.load(f)
    return model

o3_encoder = load_encoder_o3()
pm25_encoder = load_encoder_pm25()

st.set_page_config(page_title=apptitle, page_icon=':cloud:', layout='wide')

# -- Preloaded variables
valid_dates_o3 = load_data('/src/ect/valid_dates.p')
X_o3_test = load_data('/tmp/src/test/X_o3_test.p')
y_o3_test = load_data('/tmp/src/test/y_o3_test.p')
X_pm25_test = load_data('/tmp/src/test/X_pm25_test.p')
y_pm25_test = load_data('/tmp/src/test/y_pm25_test.p')
station_coordinates = load_data('/src/ect/location_coordinates.p')
station_coordinates = pd.DataFrame(station_coordinates).T.reset_index()
station_coordinates.columns = ['station', 'longitude', 'latitude']
station_coordinates['longitude'] = station_coordinates['longitude'].astype(float)
station_coordinates['latitude'] = station_coordinates['latitude'].astype(float)

# -- Session States
if 'last_execution' not in st.session_state:
    st.session_state.last_execution = 0
    
# -- Randomly choose a date to display
if 'random_date' not in st.session_state:
    st.session_state.random_date = valid_dates_o3[random.randint(0, len(valid_dates_o3) - 1)]

# -- Detect offline station
if 'station_off' not in st.session_state:
    st.session_state.station_off = True
X_dfs = {
    'o3': X_o3_test,
    'pm25': X_pm25_test
}
encoders_dict = {
    'o3': o3_encoder,
    'pm25': pm25_encoder
}
# -- Main 
st.title('Air Quality Forecaster 📈')
st.subheader('12-Hour Ahead PM2.5 & O3 Prediction')
st.caption('Powered by [Open-Meteo](https://open-meteo.com) and [OpenAQ](https://openaq.org)')
station_list = ['Belfast Centre', 'Bexley - Belvedere West',
       'Bexley - Slade Green Fidas', 'Birmingham A4540 Roadside',
       'Blackpool Marton', 'Bournemouth', 'Brent - Ikea',
       'Brighton Preston Park', 'Camden - Bloomsbury',
       'Canterbury - UKA00424', 'Cardiff Centre', 'Charlton Mackrell',
       'Chesterfield Loundsley Green', 'Chilbolton Observatory',
       'Coventry Allesley', 'Derry Rosemount', 'Edinburgh St Leonards',
       'Glazebury', 'Greenwich - Falconwood FDMS',
       'Greenwich - Plumstead High Street',
       'Greenwich - Westhorne Avenue', 'High Muffles', 'Honiton',
       'Hull Freetown', 'Kensington and Chelsea - North Ken',
       'Leamington Spa', 'Leeds Centre', 'Leicester University',
       'Lewisham - Honor Oak Park', 'Liverpool Speke',
       'London Harlington', 'London Hillingdon', 'London Westminster',
       'Lullington Heath', 'Manchester Piccadilly', 'Middlesbrough',
       'Narberth', 'Norwich Lakenfields', 'Oxford St Ebbes',
       'Plymouth Centre', 'Port Talbot Margam', 'Portsmouth', 'Preston',
       'Reading New Town', 'Rochester Stoke', 'Salford Eccles',
       'Sheffield Tinsley', 'Southampton Centre', 'Southend-on-Sea',
       'Southwark - Elephant and Castle', 'St Osyth',
       'Stoke-on-Trent Centre', 'Waterloo Place (The Crown Estate)',
       'Wicken Fen', 'Wigan Centre', 'Wirral Tranmere', 'Yarner Wood',
       'York Bootham']

# Select the station you want to predict
with st.sidebar:
    st.session_state.loaded = False 
    st.title('Select Content')
    tab1, tab2 = st.tabs(['Forecaster', 'Terminology and Q&A'])

    with tab1:
        st.markdown('## Select Station and Input Meteorological Data')
        
        # Station selection
        select_station = st.selectbox('Station', station_list)
        
        #Select Pollutant
        select_pollution = st.selectbox('Pollutant', ['o3', 'PM2.5'])
        if select_pollution == 'PM2.5':
            pollutant = 'pm25'
            station_df = X_pm25_test[X_pm25_test['station'] ==f'{select_station}']
            bins = [0, 11, 23, 35, 41, 47, 53, 58, 64, 70]
            cmap_list_lines = [
                (0, '#2ecc71'),    
                (11, '#2ecc71'),   
                (23, '#f1c40f'),   
                (35, '#f1c40f'),   
                (41, '#f39c12'),   
                (47, '#f39c12'),   
                (53, '#e74c3c'),   
                (58, '#e74c3c'),   
                (64, '#e74c3c'),  
                (70, '#8e44ad'),   
                (80, '#8e44ad'),   
            ]
        else:
            
            pollutant = 'o3'
            station_df = X_o3_test[X_o3_test['station'] ==f'{select_station}']
            
            cmap_list_lines = [
            (0, '#2ecc71'), 
            (33, '#2ecc71'), 
            (66, '#f1c40f'), 
            (100, '#f39c12'),
            (120, '#e74c3c'), 
            (180, '#8e44ad'), 
            (240, '#6c0a0a'),
                                ]
            bins = [0, 33, 66, 100, 120, 180, 240]
        # Date & time
        select_time_predict = st.time_input('Select a time:', step=3600, key='predict_time', value=datetime.time(hour=1))
        
        
        selected_date_predict = st.date_input('Select a date:', min_value=min(valid_dates_o3), max_value=max(valid_dates_o3), 
                                              value=st.session_state.random_date)
        
        predict_datetime = datetime.datetime.combine(selected_date_predict, select_time_predict)
        
        if predict_datetime.isoformat() not in station_df.index:
            st.session_state.station_off = True
            st.error('Selected date/time is unavailable due to station downtime. Please pick a valid date/time.')
            row = station_df.iloc[random.randint(0, len(station_df) - 1)] # If the date+time combination is not in the DB then simply randomly autofill
        else:
            st.session_state.station_off = False
            row = station_df.loc[predict_datetime.isoformat()]
            
        # Meteorological variables
        with st.expander('Meteorlogical Data'):
            select_t = st.number_input(f'{pollutant.upper()} Concentration (μg/m³)', float(0), float(300), 
                                       step=0.1, value=float(np.expm1(row[f'{pollutant}'])))
            select_temp = st.slider('Temperature (°C)', -10, 45, value=int(row['temperature_2m']))
            select_sp = st.number_input('Surface Pressure (hPa)', min_value=float(900), max_value=float(1100), 
                                        step=0.01, value=float(row['surface_pressure']))        
            select_pressure_msl = st.number_input('Mean Sea Level pressure (hPa)', min_value=float(900), max_value=float(1300), 
                                                  step=0.01, value=float(row['pressure_msl']))
            select_wind_speed = st.slider('Wind Speed (m/s)', float(0), float(70), step=0.01, value=float(row['wind_speed_10m']))
            select_wind_direction = st.slider('Wind Direction (°)', float(0), float(360), step=0.01, value=float(row['wind_direction_10m']))
            select_rh = st.slider('Relative Humidity (%)', 0, 100, value=int(row['relative_humidity_2m']))
            select_precip = st.slider('Precipitation (mm)', 0.0, 50.0, value=row['precipitation'])
            select_rain = st.slider('Rain mm (inch)', float(0), float(2), step=0.01, value=float(row['rain']))
            select_shortwave_radiation = st.slider('Shortwave Radiation (W/m²)', float(0), float(1200), step=0.01, 
                                                   value=float(row['shortwave_radiation']))
        selected_station_coords = station_coordinates[station_coordinates['station'] == select_station]
        #Hidden advance menu for lags/rolling features
        
        with st.expander('Advanced Settings'):
            # Hidden menu for lags, keep lags folded up to prevent it being messy
            with st.expander('Lags'):
                lags = {}
                for lag in [1, 2, 3, 6, 12, 24]:
                    lags[lag] = st.number_input(f'Lag {lag}', value=np.expm1(row[f'{pollutant}_lag_{lag}']), key=f'lag_{lag}')

            #Likewise for rollings, no need to have it showing unless needed
            values = {}
            for feature in ['mean', 'std', 'min', 'max']:
                with st.expander(f'Rolling {feature.capitalize()}'):
                    
                    for roll in [1, 2, 3, 6, 12, 24]:
                        if feature == 'mean':
                            values[f'{feature}_{roll}'] = st.number_input(
                                f'Rolling {roll}', value=row[f'{pollutant}_rolling_{roll}'], key=f'rolling_{feature}_{roll}'
                            )
                        else:
                            values[f'{feature}_{roll}'] = st.number_input(
                                f'Rolling {roll}', value=row[f'{pollutant}_rolling_{feature}_{roll}'], key=f'rolling_{feature}_{roll}'
                            )

        # Given information, start predicting 
        if st.button('Predict'):
            with st.spinner("Predicting...", show_time=True): # Make user think the model it's doing big things lmao
                time.sleep(3)
            current_time = time.time()
            time_since_clicked = current_time - st.session_state.last_execution
            plus_12 = predict_datetime + datetime.timedelta(hours=12)
            if time_since_clicked >= cooldown:
                st.session_state.loaded = True
                st.session_state.last_execution = current_time
                st.success(f'Predicting {select_pollution} for {select_station}')
                st.write(f'Date/Time: {selected_date_predict} {select_time_predict}')
                st.write(f'Temperature: {select_temp}°C')
                st.write(f'Wind Speed: {select_wind_speed} m/s')

                input_dict = {
                    'select_station': select_station,
                    'select_t': np.log1p(select_t),
                    'select_wind_speed': select_wind_speed,
                    'select_wind_direction': select_wind_direction,
                    'select_temp': select_temp,
                    'select_rh': select_rh,
                    'select_precip': select_precip,
                    'select_rain': select_rain,
                    'select_sp': select_sp,
                    'select_pressure_msl': select_pressure_msl,
                    'select_shortwave_radiation': select_shortwave_radiation,
                    'is_holiday': row['is_holiday'],
                    'day_of_week': row['day_of_week'],
                    'hour': select_time_predict.hour,
                    'sin_hour': np.sin(select_time_predict.hour),
                    'cos_hour': np.cos(select_time_predict.hour),
                    'wind_rh': select_wind_speed * select_rh,
                    'sin_wind_dir': np.sin(select_wind_direction),
                    'cos_wind_dir': np.cos(select_wind_direction),
                    'is_weekend': selected_date_predict.weekday() >= 5
                }

                # Add lags
                for interval in [1, 2, 3, 6, 12, 24]:
                    input_dict[f'lag_{interval}'] = np.log1p(lags[interval])

                # Add rolling features
                for feature in ['mean', 'std', 'min', 'max']:
                    for interval in [1, 2, 3, 6, 12, 24]:
                        input_dict[f'{feature}_{interval}'] = values[f'{feature}_{interval}']

                input_df = pd.DataFrame([input_dict])
                test_dict = {
                    
                }
                input_df.columns = X_dfs[pollutant].columns
                transformed_input = encoders_dict[pollutant].transform(input_df)
                
                if pollutant == 'o3':
                    prediction = np.expm1(o3_model.predict(transformed_input)[0])
                else:
                    threshold = 0.25  
                    predict_prob = pm25_classifier_model.predict_proba(transformed_input)[:, 1]
                    prediction = np.expm1(predict_prob * pm25_upper_model.predict(transformed_input) + (1 - predict_prob) * pm25_middle_model.predict(transformed_input))[0]
                if st.session_state.station_off:
                    st.warning('There is no data available for the selected date and time, a prediction will be made however no data will be shown on the graph.')
                else:
                    st.metric(
                        label=f'Predicted {pollutant} Concentration (µg/m³) at {plus_12.time()}, {plus_12.date()}',
                        value=f'{prediction:.2f}'
                )
                
            else:
                remaining_time = cooldown - time_since_clicked
                st.warning(f'Please wait for {remaining_time:.0f} seconds for your next prediction.')

    with tab2:
        with st.expander('Metorlogy Terminology'):
            st.markdown(" - Wind Speed: Wind speed at 10m above ground")
            st.markdown(" - Wind Direction: Wind direction at 10m above ground")
            st.markdown(" - Temperature: Air temperature at 2m above ground")
            st.markdown(" - Relative Humidity: Relative humidity at 2m above ground")
            st.markdown(" - Precipitation: Total precipitation (rain + showers + snow)")
            st.markdown(" - Rain: Rain component only")
            st.markdown(" - Surface Pressure: Atmospheric pressure at ground level")
            st.markdown(" - Sea Level Pressure: Pressure reduced to mean sea level")
            st.markdown(" - Shortwave Radiation: Global horizontal irradiance (solar radiation)")

            st.text("")
            st.markdown(" - Wind Direction (sin, cos): Cyclical encoding of wind direction")
            st.markdown(" - O₃ Lag Features (1h, 2h, 3h, 6h, 12h, 24h): Shifted values of O₃")
            st.markdown(" - O₃ Rolling Mean (1h–24h): Rolling averages of O₃")
            st.markdown(" - O₃ Rolling Std (1h–24h): Rolling standard deviation of O₃")
            st.markdown(" - O₃ Rolling Min/Max (1h–24h): Rolling min/max of O₃")
            st.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
                list-style-position: inside;
            }
            </style>
            ''', unsafe_allow_html=True)

        with st.expander('What is PM2.5?'):
            st.text("PM2.5 stands for fine particulate matter that has a diameter of 2.5 micrometers or smaller. To give you an example, it's about 30 times smaller than the width of a human hair! \n Given its small size, these particles can stay suspended in the air for long periods of time and can penetrate our lungs and even enter our bloodstream. \n They are a major pollutant and usually come from:")
            st.markdown('- Combustion processes (car engines, power plants, wood burning, industrial emissions)')
            st.markdown('- Secondary formation from chemical reactions in the atmosphere')
            st.markdown('- Natural sources (wildfires, dust, sea spray in some cases)')
            st.markdown('''
            <style>
            [data-testid="stMarkdownContainer"] ul{
                list-style-position: inside;
            }
            </style>
            ''', unsafe_allow_html=True)

        with st.expander('What is O₃?'):
            st.text("O₃ is ground-level ozone, a harmful gas that forms when sunlight reacts with nitrogen oxides (NOₓ) and volatile organic compounds (VOCs) from traffic, industry, and solvents. When it's near the ground it can harm lungs and plants.")
            st.text('Here is how it can affect people and nature:')
            st.markdown('- Health: Breathing ozone can cause coughing, reduced lung function, and worsening asthma and other respiratory problems. It is also linked to increased hospital visits and premature deaths. Outdoor workers and older adults with lung disease are most at risk.')
            st.markdown('- Environment: Ozone damages leaves, reduces photosynthesis and crop yields, and harms forest biodiversity.')

        with st.expander('Why are so many of the stations down?'):
            st.text('Stations can be offline for many reasons, for example maintenance or faulty equipment. The stations are owned by multiple providers, and it is out of OpenAQ\'s hands to maintain good uptime for each respective station.')

        with st.expander("Why can I only predict up to 15/07/2025?"):
            st.text("Because that is when I called the APIs and collected the data, making that date the cutoff point. The stations from OpenAQ are too unstable and poorly maintained and therefore not worth the hassle. \n If it wasn\'t obvious already, the data is not live.")

        with st.expander("What is the persistance model?"):
            st.text("The persistence model is one of the simplest models used in time series forecasting, which also includes the prediction of PM2.5 and O₃. Formally it can be written as:")
            st.latex(r'\hat{y}_{t} + \Delta{t} = y_{t}')
            st.text('Where:')
            st.markdown(r"""
            $$
            y_{t} = \text{The observed pollution at time} \space t \newline
            \hat{y}_{t} + \Delta{t} = \text{The pollution observed in the future at time} \space \Delta{t} + t
            $$ 
            """)
            st.text("In our case the persistence model can be written as:")
            st.latex(r'\hat{y}_{t} = y_{t + 12}')
            st.text('For those who are not too math savvy, it simply means we expect the pollution to be the same 12 hours ahead in time.')

# -- Get a DF of the stations for the selected date and hour
get_current_stations = X_dfs[pollutant].xs(predict_datetime.isoformat())[['station', 'wind_direction_10m', f'{pollutant}']]
station_coordinates_joined = pd.merge(station_coordinates, get_current_stations, on='station', how='left')
station_coordinates_joined[f'{pollutant}'] = np.expm1(station_coordinates_joined[f'{pollutant}'])
mask = station_coordinates_joined[f'{pollutant}'].isna()
station_coordinates_joined.loc[mask, 'status'] = 'Offline'
station_coordinates_joined.loc[~mask, 'status'] = 'Online'
station_coordinates_joined[f'{pollutant}'] = station_coordinates_joined[f'{pollutant}'].round(2)

    
# -- Main section of the app
col= st.columns((2, 4.5, 2), gap='medium')

station_coordinates_joined_style = station_coordinates_joined.style.applymap(status_color, subset=["status"])

with col[2]:
    st.markdown('#### Station Status')

    with st.expander('Status', expanded=True):
        st.dataframe(station_coordinates_joined_style,
                    column_order=("station", "status"),
                    hide_index=True,
                    column_config={
                        "station": st.column_config.TextColumn(
                            "Stations",
                        ),
                        "status": st.column_config.TextColumn(
                        "Status"
                        )}
                    )

# -- This session state ensures that the plot only occurs when the user has pressed the prediction button, 
# -- otherwise nothing will show up.
with col[1]:
    station_coordinates_joined.loc[mask, 'station'] = station_coordinates_joined.loc[mask, 'station'] + '\n (Offline)'
    placeholder = st.empty()
    if not st.session_state.loaded:
        empty_df = pd.DataFrame(columns=['Time', f'{pollutant}', 'type'])
        chart = (
            alt.Chart(empty_df)
            .mark_line(point=True)
            .encode(
            x='Time:T',
            y=f'{pollutant}:Q',
            color='type:N'
            )
        )
        placeholder.altair_chart(chart, use_container_width=True)

    if st.session_state.loaded:
        
        end_interval = predict_datetime + datetime.timedelta(hours=13)

        data = np.expm1(station_df.loc[predict_datetime.isoformat():end_interval.isoformat()][f'{pollutant}']) # Need to make sure to remove log transformation so it's clearer for the user
        data = pd.DataFrame(data, index=data.index).reset_index()
        data.columns = ['Time', f'{pollutant}']
        plus_12 = predict_datetime + datetime.timedelta(hours=12)
        data['type'] = 'Ground Truth'
        line = (
            alt.Chart(data)
            .mark_line(point=True)
            .encode(
            x='Time:T',
            y=f'{pollutant}:Q',
            color='type:N'
            )
        )

        predicted_point = pd.DataFrame(
            {'Time': [pd.Timestamp(plus_12)],
            f'{pollutant}': prediction,
            'type': ['Model Prediction']}
        )
        persistance_point = pd.DataFrame(
            {'Time': [pd.Timestamp(plus_12)],
            f'{pollutant}': np.expm1(row[f'{pollutant}']),
            'type': ['Persistance Prediction']}
        )
        point = (
            alt.Chart(predicted_point)
            .mark_point(color='red')
            .encode(
            x='Time:T',
            y=f'{pollutant}:Q',
            color='type:N'
            )
        )
        persistance = (
            alt.Chart(persistance_point)
            .mark_point(color='yellow')
            .encode(
            x='Time:T',
            y=f'{pollutant}:Q',
            color='type:N'
            )
        )


        chart = (
        line + point + persistance
        ).properties(
            width='container',
            height=400  # increase if it's cutting off the bottom
        ).configure_view(
            stroke=None  # removes the default gray border that can hide parts
        )

        st.altair_chart(chart)

    # -- End of session state loaded
    layers = []
    show_pins, show_wind_dir, show_pollution = st.columns([1, 1, 1])
    with show_pins:
        show_pins_checked = st.checkbox('Show Stations', value=True)
    with show_wind_dir:
        show_wind_dir_checked = st.checkbox('Show Wind Direction', value=True)
    with show_pollution:
        show_pollution_checked = st.checkbox('Show Pollution', value=True)




    cmap_lines, norm_lines = cmap_continuous(cmap_list_lines)
    colors = [rgb2hex(cmap_lines(norm_lines(v))) for v in station_coordinates_joined[f'{pollutant}'].values]
    station_coordinates_joined['color'] = [
        list(mcolors.to_rgba(c, alpha=0.8))[:3] + [200]  # [R,G,B,Alpha]
        for c in colors
    ]
    station_coordinates_joined['color'] = station_coordinates_joined['color'].apply(lambda x: [int(c * 255) for c in x])
    cmap = ListedColormap(station_coordinates_joined['color'].values)



    icon_data = {
        'url': 'https://maps.gstatic.com/mapfiles/api-3/images/spotlight-poi2.png',
        'width': 128,
        'height': 128,
        'anchorY': 128,
    }

    arrow_icon = {
        'url': green_arrow,
        'width': 128,
        'height': 128,
        'anchorY': 64,
        'anchorX': 64 
    }

    station_coordinates_joined['icon_pin'] = [icon_data for _ in range(len(station_coordinates_joined))]
    station_coordinates_joined['latitude_scale'] = station_coordinates_joined['latitude'] - 0.0006

    view_state = pdk.ViewState(
        latitude=float(selected_station_coords['latitude']), longitude=float(selected_station_coords['longitude']), controller=True, zoom=14, pitch=0,

    )

    station_coordinates_joined['icon_arrow'] = [arrow_icon for _ in range(len(station_coordinates_joined))]
    if show_pollution_checked:
        pollution_layer = pdk.Layer(
            'ScatterplotLayer',
            station_coordinates_joined.dropna(subset=[f'{pollutant}']),
            get_position=['longitude', 'latitude'],
            get_radius=f'{pollutant}',
            radius_scale=50,  # Adjust to make circles appropriate size
            radius_min_pixels=20,  # Minimum circle size
            radius_max_pixels=100,  # Maximum circle size
            get_fill_color='color',
            pickable=True,
            stroked=True,
            get_line_color=[255, 255, 255],
            line_width_min_pixels=1,
            opacity=0.4
        )
        layers.append(pollution_layer)
        
    if show_wind_dir_checked:
        wind_layer = pdk.Layer(
            'IconLayer',
            data=station_coordinates_joined[station_coordinates_joined['wind_direction_10m'].notna()],
            get_icon='icon_arrow',
            get_size=20,
            get_position=['longitude', 'latitude_scale'],
            get_angle='wind_direction_10m',
            opacity=1
        )
        layers.append(wind_layer)
    if show_pins_checked:
        point_layer = pdk.Layer(
            'IconLayer',
            data=station_coordinates_joined,
            get_icon='icon_pin',
            get_size=40,
            get_position=['longitude', 'latitude'],
            pickable=True,
            # auto_highlight=True,
            # get_radius=200,
            
        )
        layers.append(point_layer)




    map_col, cbar_col = st.columns([4, 1]) 

    with map_col:
        event = st.pydeck_chart(
            pdk.Deck(
                layers,
                initial_view_state=view_state,
                tooltip={'text': 'Station: {station}' f'\n Pollution: {{{pollutant}}}'},
            ),
            on_select='rerun', 
            selection_mode='multi-object'
        )

    with cbar_col:

        fig, ax = plt.subplots(figsize=(1.5, 3))
        ax.set_visible(False)
        
        sm = cm.ScalarMappable(cmap=cmap_lines, norm=norm_lines)
        sm.set_array([])
        
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label(f'{pollutant.upper()} (μg/m³)', rotation=270, labelpad=20)
        
        cbar.set_ticks(bins)
        cbar.set_ticklabels(bins)
        cbar.ax.tick_params(colors='white')  
        cbar.ax.yaxis.label.set_color('white')  
        cbar.outline.set_edgecolor('white') 
        st.pyplot(fig, transparent=True)
        plt.close()
        
    event.selection

    