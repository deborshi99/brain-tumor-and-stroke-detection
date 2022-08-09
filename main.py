import streamlit as st
from streamlit_option_menu import option_menu
from utils.analysis import train_data, scaler
import pandas as pd
import shutil
import os
import tensorflow as tf
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
from utils.constant import brain_tumor_stoke_model, tumor_or_normal
from keras.utils import load_img, img_to_array
from random import randint

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
        #st.markdown(Style,unsafe_allow_html=True)

        file = st.file_uploader("Please upload image/images", type=["png", "jpeg", "jpg"], accept_multiple_files=True)
        show_file = st.empty()

        if not file:
                show_file.info("Please Upload a image in {} format".format(", ".join(["png", "jpeg", "jpg"])))
        else:
                def predict():
                        pred = {}
                        os.makedirs("test_data", exist_ok=True)
                        images = file
                        for i in images:
                                img = Image.open(i)
                                img = ImageOps.grayscale(img)
                                img.save(f"test_data/{i.name}")
                        
                        image_path = [os.path.join("test_data", fname) for fname in os.listdir("test_data")]
                        #st.write(image_path)
                        if len(image_path) > 1:
                                return 1, image_path
                        else:
                                return 0, image_path

                result, data = predict()
                if result == 1:
                        if len(file)>1:
                                show_file.success("Files Uploaded Successfully")
                        else:
                                show_file.success("File Uploaded Successfully")
                
                def predict_image(file_path):
                        results = {}        
                        for i in file_path:
                                img = load_img(i, target_size=(150, 150), color_mode="grayscale")
                                x = img_to_array(img)
                                x /= 255
                                x = np.expand_dims(x, axis=0)

                                images = np.vstack([x])
                                classes = tumor_or_normal.predict(images, batch_size=10)
                                labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
                                if classes[0]<0.5:
                                        st.spinner("Detecting type of tumor")
                                        tumor_classes = brain_tumor_stoke_model.predict(images)
                                        pred = np.argmax(tumor_classes)
                                        if pred == 0:
                                                results[i.split("/")[-1]] = labels[0]
                                        elif pred == 1:
                                                results[i.split("/")[-1]] = labels[1]
                                        elif pred == 2:
                                                results[i.split("/")[-1]] = labels[2]
                                        elif pred == 3:
                                                results[i.split("/")[-1]] = labels[3]                                       

                                else:
                                        results[i.split("/")[-1]] = "Cancer is not detected"
                        return results
                
                
                result = predict_image(data)
                shutil.rmtree("test_data")

                if st.button("Predict"):
                        if len(result)>=1:
                                def show_output(result, data):
                                        col1, col2, col3 = st.columns(3)
                                        for i in data:
                                                col1.image(i, caption=f"Image name: {i.name}", width=200)

                                                col1.info(f"{result[i.name]} tumor has been detected")
                                        
                                
                                show_output(result, file)






                        
                
                
                



elif selected == "Brain Stroke Possibility":

        ###################### input data #####################################3
        col1, col2 = st.columns(2)

        gender = col1.selectbox("Select your gender", options=["Female", "Male"])
        age = col1.number_input("Enter the age")
        hypertension = col1.selectbox("Do you have hypertension", options=["Yes", "No"])
        heart_disease = col1.selectbox("Do you have any heart disease", options=["Yes", "No"])
        ever_married = col1.selectbox("Were/Are you married", options=["Yes", "No"])
        work_type = col2.selectbox("Select your work type", options=["Private", "Self-employed", "Children", "Govt_job"])
        resident_type = col2.selectbox("Where do you live", options=["Urban", "Rural"])
        avg_glucose_level = col2.number_input("Please enter your average glucose level")
        bmi = col2.number_input(f"Please enter your BMI {st.write('To calculate BMI, click [here](https://www.nhlbi.nih.gov/health/educational/lose_wt/BMI/bmicalc.htm)')}")
        smoke_status = col2.selectbox("Do/Did you smoke", options=["never smoked", "Unknown", "formerly smoked", "smokes"])
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
                        st.success(result)
        elif y_pred == 1:
                if st.button("Predict"):
                        result = "your chances are high of getting a brain stroke, kindly consult a doctor"
                        st.success(result)

        




