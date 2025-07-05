# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# === Load Dataset ===
@st.cache_data
def load_data():
    df = pd.read_csv("audi.csv")
    return df

# === Train Model ===
@st.cache_resource
def train_model(df):
    x = df[['year', 'mileage', 'tax', 'mpg', 'engineSize']]
    y = df['price']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

# === Streamlit UI ===
def main():
    st.title("Prediksi Harga Mobil Bekas (Ford)")
    st.write("Masukkan spesifikasi mobil Anda:")

    year = st.number_input("Tahun Produksi", min_value=1990, max_value=2025, value=2017)
    mileage = st.number_input("Jarak Tempuh (dalam mil)", min_value=0, value=20000)
    tax = st.number_input("Pajak (£)", min_value=0, value=150)
    mpg = st.number_input("Efisiensi Bahan Bakar (mpg)", min_value=0.0, value=55.0)
    engineSize = st.number_input("Kapasitas Mesin (L)", min_value=0.0, value=1.6)

    if st.button("Prediksi Harga"):
        df = load_data()
        model = train_model(df)

        input_data = pd.DataFrame({
            'year': [year],
            'mileage': [mileage],
            'tax': [tax],
            'mpg': [mpg],
            'engineSize': [engineSize]
        })

        prediction = model.predict(input_data)[0]
        st.success(f"Perkiraan harga mobil bekas Anda adalah: £{prediction:,.2f}")

if __name__ == '__main__':
    main()
