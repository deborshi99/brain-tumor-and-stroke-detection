import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from keras import Sequential
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

dataset = pd.read_csv("dataset/brain_stroke.csv")

dataset["hypertension"] = dataset["hypertension"].values.astype(str)
dataset["heart_disease"] = dataset["heart_disease"].values.astype(str)

train_data = dataset.drop("stroke", axis=1)
train_label = dataset["stroke"]

oversample = RandomOverSampler(sampling_strategy="minority")
X, y = oversample.fit_resample(train_data, train_label)

train_data_dummy = pd.get_dummies(X)

scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data_dummy)

X_train, X_test, y_train, y_test = train_test_split(train_data_scaled, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


