import streamlit as st
import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
import gc
import time

o3_encoder = None
pm25_encoder = None
encode_dict = {
    'pm25': pm25_encoder,
    'o3': o3_encoder
}

# ============================================================================
# LOAD MODELS
# ============================================================================
@st.cache_resource
def load_model_o3():
    model = xgb.XGBRegressor()
    model.load_model('/mount/src/pollution_pred/pollution_data/models/xg_reg_o3.model')
    return model 

@st.cache_resource
def load_model_pm25_classifier():
    model = xgb.XGBClassifier()
    model.load_model('/mount/src/pollution_pred/pollution_data/models/binary_classifier.model')
    return model 

@st.cache_resource
def load_model_pm25_middle():
    model = xgb.XGBRegressor()
    model.load_model('/mount/src/pollution_pred/pollution_data/models/xg_mid.model')
    return model 

@st.cache_resource
def load_model_pm25_upper():
    model = xgb.XGBRegressor()
    model.load_model('/mount/src/pollution_pred/pollution_data/models/xg_90.model')
    return model 

@st.cache_resource
def load_encoder_o3():
    with open('/mount/src/pollution_pred/pollution_data/models/encoder_o3.model', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_encoder_pm25():
    with open('/mount/src/pollution_pred/pollution_data/models/encoder_pm25.model', 'rb') as f:
        model = pickle.load(f)
    return model

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def make_prediction(select_pollution, rolling_data, input_dict, lags, test_dataframes):
    if select_pollution == 'pm25':
        pm25_classifier_model = load_model_pm25_classifier()
        pm25_middle_model = load_model_pm25_middle()
        pm25_upper_model = load_model_pm25_upper()
        pm25_encoder = load_encoder_pm25()
        pm25_classifier_model.set_params(tree_method='hist', device='cpu')
        pm25_middle_model.set_params(tree_method='hist', device='cpu')
        pm25_upper_model.set_params(tree_method='hist', device='cpu')
        encode_dict['pm25'] = pm25_encoder
    else:
        o3_encoder = load_encoder_o3()
        o3_model = load_model_o3()
        encode_dict['o3'] = o3_encoder
        o3_model.set_params(tree_method='hist', device='cpu')
        
    # Add lags
    for interval in [1, 2, 3, 6, 12, 24]:
        input_dict[f'lag_{interval}'] = np.log1p(lags[interval])

    # Add rolling features
    for feature in ['mean', 'std', 'min', 'max']:
        for interval in [1, 2, 3, 6, 12, 24]:
            input_dict[f'{feature}_{interval}'] = rolling_data[f'{feature}_{interval}']

    input_df = pd.DataFrame([input_dict])
    input_df.columns = test_dataframes[select_pollution].columns
    transformed_input = encode_dict[select_pollution].transform(input_df)
    
    if select_pollution == 'o3':
        prediction = np.expm1(o3_model.predict(transformed_input)[0])
        del o3_encoder; gc.collect()
        del o3_model; gc.collect()
    else:
        predict_prob = pm25_classifier_model.predict_proba(transformed_input)[:, 1]
        prediction = np.expm1(predict_prob * pm25_upper_model.predict(transformed_input) + (1 - predict_prob) * pm25_middle_model.predict(transformed_input))[0]
        del pm25_classifier_model; gc.collect()
        del pm25_encoder; gc.collect()
        del pm25_middle_model; gc.collect()
        del pm25_upper_model; gc.collect()
    return prediction