import streamlit as st
import os
import gdown

def download_models_and_data():
    
    if os.path.exists('./pollution_data'):
        print("Files already exist, skipping download.")
        return
    model_files = {
        "1nCj6zLMocxIT-mGA7gqIB0tqCs_A3DpW": "binary_classifier.model",
        "1siqY-_o2j9zyVH_ouXZHezsq7Sa1OVaK": "encoder_o3.model",
        "1qpdnUVErI2qNh0BrvPXsXMuF9FBs8EDH": "encoder_pm25.model",
        "1z2q62ZbcvJZ1UnBctHKMWvmoxjiC0A3k": "xg_90.model",
        "1LcUqJCRge7yxGOD9Q9w-lnqoVa8ilurm": "xg_mid.model",
        "1zJO2hvk_kaTE3kt-glXcv_Br2Sx39Z5A": "xg_reg_o3.model"
    }
    progress_bar_1 = st.progress(0, text="Downloading models...")
    progress_bar_2 = st.progress(0, text="Downloading test files...")
    output_test = "./pollution_data/models"
    os.makedirs(output_test, exist_ok=True)
    for i, (file_id, filename) in enumerate(model_files.items()):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output=os.path.join(output_test, filename), quiet=False)
        progress_bar_1.progress((i+1)/len(model_files), text=f"Downloading models ({i+1}/{len(model_files)})")
        
    test_files = {
        "1dD5CohGL9jp3y_kCGKwEHkB2pMkXqSqu": "X_o3_test.p",
        "1amy0T8czFfJjJBQ6rjzqofjwcZDuWF_0": "y_o3_test.p",
        "1TUfv052yVfr3qr34wsETZM8gFGifjl27": "X_pm25_test.p",
        "1nc7YtacKPwpUFTX1LzZFUKHqAzcG35Li": "y_pm25_test.p",
    }

    output_test = "./pollution_data/test"
    os.makedirs(output_test, exist_ok=True)
    for i, (file_id, filename) in enumerate(test_files.items()):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output=os.path.join(output_test, filename), quiet=False)
        progress_bar_2.progress((i+1)/len(test_files), text=f"Downloading test files ({i+1}/{len(test_files)})")
    progress_bar_1.empty()
    progress_bar_2.empty()
    
if __name__ == "__main__":
    download_models_and_data()