from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report
import streamlit as st
import joblib
import pandas as pd


st.header("Breast Cancer Classification", divider='rainbow')


selected = option_menu(
  menu_title="",
  options=["Training Data", "Klasifikasi"],
  icons=["None", "None"],
  orientation="horizontal"
  )

# ===================== Halaman Training =====================
if selected == "Training Data":
    dataset, ta, tt = st.tabs(["Dataset", "Hasil Skenario Training", "Try Train"])
    with dataset:
        df = pd.read_csv("resources/data normalisasi.csv")
        df = df.drop(["Unnamed: 0"], axis=1)
        st.write("Dataset yang digunakan (Telah Di Normalisasi):")
        st.dataframe(df, use_container_width=True)
    
    with ta:
        st.write("Skenario Hasil Uji Coba :")
        df = pd.read_csv("resources/skenario.csv")
        df = df.drop(["Unnamed: 0"], axis=1)
        st.dataframe(df, use_container_width=True)
    
    with tt:
        x = pd.read_csv("resources/data normalisasi.csv")
        x_norm = x.drop("diagnosis_M", axis=1)
        y = x.iloc[:,-1]
        test_size = st.slider("tes_size", 0.0, 1.0, value=0.2)
        n_tree = st.number_input("N Tree", 1, 10000, value=200)
        button = st.button("Train")
        if button:
            x_train, x_test, y_train, y_test = train_test_split(x_norm,y, test_size=test_size, random_state=0)
            custom_model = RandomForestClassifier(n_estimators=n_tree, random_state=0)
            custom_model.fit(x_train, y_train)
            y_pred = custom_model.predict(x_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            
            if "result" in st.session_state:
                result = st.session_state.result
            
            else:
                result = pd.DataFrame([])
                
            model_result = pd.DataFrame([[f'Model With {n_tree} Decission Tree', acc, f1, prec, rec]], columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])
            result = result.append(model_result, ignore_index=True)
            st.session_state.result = result
            st.write("History Training: ")
            st.dataframe(st.session_state.result, use_container_width=True)
            
            
        
# ===================== Halaman Klasifikasi =====================
elif selected == "Klasifikasi":
    st.caption("Model ini menggunakan 200 N Tree sebagai model predict")
    col1, col2, col3 = st.columns(3)
    with col1:
        radius_m = st.slider("Raidus Mean", 0.0, 20.590000, value=10.0)
        texture_m = st.slider("Texture Mean", 0.0, 29.810000, value=20.0)
        primeter_m = st.slider("Primeter Mean", 0.0, 137.800000, value=40.0)
        area_m = st.slider("Area Mean", 0.0, 1320.000000, value=100.0)
        smoothness_m = st.slider("Smoothness Mean", 0.0, 0.125700, value=0.02)
        compactness_m = st.slider("Compactness Mean", 0.0, 0.202200, value=0.1)
        covacity_m = st.slider("Covacity Mean", 0.0, 0.254500, value=0.1)
        convave_m = st.slider("Convave Point Mean", 0.0, 0.125900, value=0.01)
        simmetry_m = st.slider("Simmetry Mean", 0.0, 0.245900, value=0.02)
        fractal_m = st.slider("Fractal Dimention Mean", 0.0, 0.078180, value=0.003)
    
    with col2:
        radius_se = st.slider("Raidus Se", 0.0, 0.747400, value=0.2)
        texture_se = st.slider("Texture Se", 0.0, 2.426000, value=0.2)
        primeter_se = st.slider("Primeter Se", 0.0, 5.216000, value=0.2)
        area_se = st.slider("Area Se", 0.0, 83.500000, value=20.0)
        smoothness_se = st.slider("Smoothness Se", 0.0, 0.012150, value=0.002)
        compactness_se = st.slider("Compactness Se", 0.0, 0.055920, value=0.002)
        covacity_se = st.slider("Covacity Se", 0.0, 0.081580, value=0.02)
        convave_se = st.slider("Convave Point Se", 0.0, 0.022580, value=0.002)
        simmetry_se = st.slider("Simmetry Se", 0.0, 0.035040, value=0.002)
        fractal_se = st.slider("Fractal Dimention Se", 0.0, 0.008015, value=0.0002,  step=0.00001)
    
    with col3:
        radius_worst = st.slider("Raidus Worst", 0.0, 24.560000, value=5.0)
        texture_worst = st.slider("Texture Worst", 0.0, 40.540000, value=10.0)
        primeter_worst = st.slider("Primeter Worst", 0.0, 166.400000, value=20.0)
        area_worst = st.slider("Area Worst", 0.0, 1872.000000, value=200.0)
        smoothness_worst = st.slider("Smoothness Worst", 0.0, 0.187800, value=0.01)
        compactness_worst = st.slider("Compactness Worst", 0.0, 0.611000, value=0.2)
        covacity_worst = st.slider("Covacity Worst", 0.0, 0.772700, value=0.2)
        convave_worst = st.slider("Convave Point Worst", 0.0, 0.254300, value=0.02)
        simmetry_worst = st.slider("Simmetry Worst", 0.0, 0.412800, value=0.2)
        fractal_worst = st.slider("Fractal Dimention Worst", 0.0, 0.120500, value=0.02)
    
    model = joblib.load("resources/model rf.pkl")
    input_data  = [
        [
            radius_m, 
            texture_m,
            primeter_m, 
            area_m,
            smoothness_m,
            compactness_m,
            covacity_m,
            convave_m,
            simmetry_m,
            fractal_m,
            
            radius_se, 
            texture_se,
            primeter_se, 
            area_se,
            smoothness_se,
            compactness_se,
            covacity_se,
            convave_se,
            simmetry_se,
            fractal_se,
            
            radius_worst, 
            texture_worst,
            primeter_worst, 
            area_worst,
            smoothness_worst,
            compactness_worst,
            covacity_worst,
            convave_worst,
            simmetry_worst,
            fractal_worst
            ]
        ]
    new_dataa = [[1.825e+01, 1.998e+01, 1.196e+02, 1.040e+03, 9.463e-02, 1.090e-01, 1.127e-01, 7.400e-02, 1.794e-01,5.742e-02, 4.467e-01, 7.732e-01, 3.180e+00, 5.391e+01, 4.314e-03, 1.382e-02, 2.254e-02, 1.039e-02, 1.369e-02, 2.179e-03, 2.288e+01, 2.766e+01, 1.532e+02, 1.606e+03, 1.442e-01, 2.576e-01, 3.784e-01, 1.932e-01, 3.063e-01, 8.368e-02]]


    predict = st.button("Predict")
    st.header("Prediction", divider='rainbow')
    if predict:
        scaler = joblib.load("resources/minmaxScaler.pkl")
        data_scaler = scaler.transform(new_dataa)
        prediction = model.predict(data_scaler)
        diagnosis = ['Jinak', 'Ganas']
        st.write(f"Hasil Prediksi : {diagnosis[prediction[0]]}")
    
