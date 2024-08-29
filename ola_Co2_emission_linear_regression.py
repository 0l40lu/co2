import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv('cars.csv')
df = data[['Car',
           'Model',
           'Volume',
           'Weight',
           'CO2']].dropna()

car_encoder = LabelEncoder()
df['Car'] = car_encoder.fit_transform(df['Car'])

model_encoder = LabelEncoder()
df['Model'] = model_encoder.fit_transform(df['Model'])

x = df[['Car',
        'Model',
        'Volume',
        'Weight',]]
y = df['CO2']

feature_train, feature_test, target_train, target_test = train_test_split(x, y, test_size=0.2)

model = LinearRegression()
model.fit(feature_train, target_train)

st.header('Co2 Emission Prediction Created by Ola')
Cars = st.sidebar.selectbox('Car', car_encoder.classes_)
Models = st.sidebar.selectbox('Model', model_encoder.classes_)
Volumes = st.sidebar.number_input('Volume', 1000, 3000)
Weights = st.sidebar.number_input('Weight', 720, 1900)

car_encode = car_encoder.transform([Cars])[0]
model_encode = model_encoder.transform([Models])[0]

total = {
    'Car': [car_encode],
    'Model': [model_encode],
    'Volume': [Volumes],
    'Weight': [Weights]
}

features = pd.DataFrame(total)
dt = pd.DataFrame(total)
st.dataframe(dt, width=500)

if st.button('Check'):
    prediction = model.predict(features)
    st.write('CO2 Emission:', prediction[0])
