from cgi import test
from genericpath import exists
import streamlit as st
from streamlit_option_menu import option_menu
from utils.analysis import train_data, scaler
import pandas as pd
import shutil
import os
import tensorflow as tf
import numpy as np
from keras.models import load_model


st.set_page_config(
page_title="Brain Disease",
page_icon="chart_with_upwards_trend",
layout="wide",
)

# Create a sidebar menu

selected = option_menu(
        menu_title=None,
        options=["Brain Tumor Detection", "Brain Stroke Possibility"],
        icons=["question-circle-fill", "activity"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
 )

if selected == "Brain Tumor Detection":
        st.write("on progress")

elif selected == "Brain Stroke Possibility":

        ###################### input data #####################################3
        gender = st.selectbox("Select your gender", options=["Female", "Male"])
        age = st.number_input("Enter the age")
        hypertension = st.selectbox("Do you have hypertension", options=["Yes", "No"])
        heart_disease = st.selectbox("Do you have any heart disease", options=["Yes", "No"])
        ever_married = st.selectbox("Were/Are you married", options=["Yes", "No"])
        work_type = st.selectbox("Select your work type", options=["Private", "Self-employed", "Children", "Govt_job"])
        resident_type = st.selectbox("Where do you live", options=["Urban", "Rural"])
        avg_glucose_level = st.number_input("Please enter your average glucose level")
        bmi = st.number_input(f"Please enter your BMI {st.write('To calculate BMI, click [here](https://www.nhlbi.nih.gov/health/educational/lose_wt/BMI/bmicalc.htm)')}")
        smoke_status = st.selectbox("Do/Did you smoke", options=["never smoked", "Unknown", "formerly smoked", "smokes"])
        #########################################################################

        data = []
        def convert(var):
                if var == "Yes":
                        var = 1
                else:
                        var = 0
                return var

        data.append(gender)
        data.append(age)
        hypertension = convert(hypertension)
        data.append(hypertension)
        heart_disease = convert(heart_disease)
        data.append(heart_disease)
        data.append(ever_married)
        data.append(work_type)
        data.append(resident_type)
        data.append(avg_glucose_level)
        data.append(bmi)
        data.append(smoke_status)

        def user_data(data):
                os.makedirs("user_data", exist_ok=True)
                data_dict = {}
                columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married','work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
                                'smoking_status']
                for i in range(len(columns)):
                        data_dict[columns[i]] = [data[i]]
                
                user_data = pd.DataFrame.from_dict(data_dict)
                user_data.to_csv("user_data/test_data.csv")
                test_data = pd.read_csv("user_data/test_data.csv")
                shutil.rmtree("user_data")
                return test_data

        test_data = user_data(data)
        test_data["hypertension"] = test_data["hypertension"].values.astype(str)
        test_data["heart_disease"] = test_data["heart_disease"].values.astype(str)
        test_data = pd.concat([train_data, test_data])

        
        data = pd.get_dummies(test_data)
        data.drop("Unnamed: 0", axis=1, inplace=True)
        data = scaler.transform(data)
        model = load_model("model/main_model.h5")
        user_input = tf.expand_dims(data[-1], axis=-1)
        user_input = np.transpose(user_input)
        y_pred = model.predict(user_input)
        y_pred = tf.round(y_pred)
        if y_pred == 0:        
                if st.button("Predict"):
                        result = "your chances are low of getting a brain stroke"
                        st.write(result)
        elif y_pred == 1:
                if st.button("Predict"):
                        result = "your chances are high of getting a brain stroke, kindly consult a doctor"
                        st.write(result)

        




