import streamlit as st
from streamlit_option_menu import option_menu
import json
import requests
import pandas as pd
import statistics
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

#-----------------------------------------------------------STREAMLIT---------------------------------------------------------------------------------------------------------
df=pd.read_csv('D:/My Projects/Singapure-Resale-flat_price -Prediction/resale_csv')

# -------------------------------This is the configuration page for our Streamlit Application----------------------------------------
st.set_page_config(
    page_title='Singapure  Resale flat price Prediction',
    page_icon="🏨",
    layout='wide'
)

#----------------------------------------This is the sidebar in a Streamlit application, helps in navigation----------------------------------
with st.sidebar:
    selected = option_menu("Main Menu", ["About Project", "Predictions"],
                           icons=["house", "gear"],
                           styles={"nav-link": {"font": "sans serif", "font-size": "20px", "text-align": "centre"},
                                   "nav-link-selected": {"font": "sans serif", "background-color": "#0072b1"},
                                   "icon": {"font-size": "20px"}
                                   })
    
# -----------------------------------------------About Project Section--------------------------------------------------
if selected == "About Project":
    st.markdown("# :blue[Singapore Resale Flat Prices Prediction]")
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown("### :blue[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building"
                "Model Deployment")
    st.markdown("### :blue[Overview :] This project aims to construct a machine learning model and implement "
                "it as a user-friendly online application in order to provide accurate predictions about the "
                "resale values of apartments in Singapore. This prediction model will be based on past transactions "
                "involving resale flats, and its goal is to aid both future buyers and sellers in evaluating the "
                "worth of a flat after it has been previously resold. Resale prices are influenced by a wide variety "
                "of criteria, including location, the kind of apartment, the total square footage, and the length "
                "of the lease. The provision of customers with an expected resale price based on these criteria is "
                "one of the ways in which a predictive model may assist in the overcoming of these obstacles.")
    st.markdown("### :blue[Domain :] Real Estate")
    st.markdown("### :red[Developed by :] Mohana Anjan A V")

    # ------------------------------------------------Predictions Section---------------------------------------------------
if selected == "Predictions":
    st.markdown("# :blue[Predicting Results based on Trained Models]")
    st.markdown("### :orange[Predicting Resale Price (Regression) (Accuracy: 95%)]")
    with st.form("form1"):
        col1,col2 = st.columns([5,5]) 
        # -----New Data inputs from the user for predicting the resale price-----

    
        with col1:
            town= st.selectbox('Select town',sorted(df.town.unique()))
            flat_type = st.selectbox('Select flat_type',sorted(df.flat_type.unique()))
            address = st.selectbox('Select Address',sorted(df.address.unique()))
            flat_model = st.selectbox('Select flat_model', sorted(df.flat_model.unique()))
            resale_year =st.selectbox('Select resale year',sorted(df.resale_year.unique()))
            resale_month = st.selectbox('Select resale month',sorted(df.resale_month.unique()))
        with col2:
            Floor_Area = st.number_input('Floor Area (Per Square Meter)', min_value=1.0, max_value=500.0)
            lease_commence_date = st.selectbox('Select lease_commence_date',sorted(df.lease_commence_date.unique()))
            storey_min = st.number_input("Storey Min (Enter the  value( 1 to 51))",min_value=1,max_value=51)
            storey_max = st.number_input("Storey Max (Enter the  value(1 to 51))",min_value=1,max_value=51)
            lease_remain_years = 99 - (2024 - lease_commence_date)
            



        # -----Submit Button for PREDICT RESALE PRICE-----
            submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")
            if submit_button is not None:
                with open(r"model.pkl", 'rb') as file:
                    loaded_model = pickle.load(file)
                with open(r'scaler.pkl', 'rb') as f:
                    scaler_loaded = pickle.load(f)
                with open(r"le_address.pkl", 'rb') as f:
                    address_loaded = pickle.load(f)
                with open(r"le_flat_model.pkl", 'rb') as f:
                    flat_model_loaded = pickle.load(f)
                with open(r"le_flat_type.pkl", 'rb') as f:
                    flat_type_loaded = pickle.load(f)
                with open(r"le_town.pkl", 'rb') as f:
                    town_loaded = pickle.load(f)
                    # -----Sending that data to the trained models for  Resale  price prediction----------------
                new_sample=np.array([[town_loaded.transform([town])[0],flat_type_loaded.transform([flat_type])[0],Floor_Area,   
                                    flat_model_loaded.transform([flat_model])[0],
                                      lease_commence_date,lease_remain_years,resale_year,resale_month,
                                      address_loaded.transform([address])[0],
                                      storey_min,storey_max
                     
                     ]])

               # Reshape to 2D array
                new_sample5 = scaler_loaded.transform(new_sample)
                new_pred = loaded_model.predict(new_sample5)[0]
                st.write('## :green[Predicted Resale  price:] ', new_pred)

                    
