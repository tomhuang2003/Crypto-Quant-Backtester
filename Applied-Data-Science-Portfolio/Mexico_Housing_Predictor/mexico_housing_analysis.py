import warnings
from glob import glob
import streamlit as st
import os

import pandas as pd
from category_encoders import OneHotEncoder
from ipywidgets import Dropdown, FloatSlider, IntSlider, interact
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline


warnings.simplefilter(action="ignore", category=FutureWarning)


def wrangle(filepath):
    df = pd.read_csv(filepath)

    mask_apt = df["property_type"] == "apartment"
    mask_cose = df["price_aprox_usd"] < 100000
    mask_ba = df["place_with_parent_names"].str.contains("Distrito Federal")
    df = df[mask_apt & mask_cose & mask_ba]

    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_quantile = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_quantile]

    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)

    df["borough"] = df["place_with_parent_names"].str.split("|", expand=True)[1]
    df.drop(columns="place_with_parent_names", inplace=True)

    df.drop(columns=[
        "surface_total_in_m2", "price_usd_per_m2", "floor", "rooms", "expenses",
        "operation", "property_type", "currency", "properati_url", "price",
        "price_aprox_local_currency", "price_per_m2"
    ], inplace=True)

    return df

base_path = os.path.dirname(os.path.abspath(__file__))
files = glob(os.path.join(base_path, "data", "mexico-city-real-estate-*.csv"))
frames = (wrangle(file) for file in files)
df = pd.concat(frames, ignore_index=True)


feature = ["surface_covered_in_m2", "lat", "lon", "borough"]
target = "price_aprox_usd"
X_train = df[feature]
y_train = df[target]


y_mean = y_train.mean().round(2)
y_pred_baseline = [y_mean] * len(y_train)
baseline_mae = round(mean_absolute_error(y_train, y_pred_baseline),2)
print("Mean apt price:", y_mean)
print("Baseline MAE:", baseline_mae)

model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(),
    Ridge(),
)
model.fit(X_train, y_train)

y_pred_training = model.predict(X_train)
print("Training MAE:", mean_absolute_error(y_train,y_pred_training))

X_test = pd.read_csv(os.path.join(base_path, "data", "mexico-city-test-features.csv"))
y_test_pred = pd.Series(model.predict(X_test))

coefficients = model.named_steps["ridge"].coef_
features = model.named_steps["onehotencoder"].get_feature_names()
feat_imp = pd.Series(coefficients, index=features).sort_values(key=abs)


def make_prediction(area, lat, lon, borough):
    data = {
        "surface_covered_in_m2":area,
        "lat":lat,
        "lon":lon,
        "borough": borough
    }
    X_test = pd.DataFrame(data,index=[0])
    prediction = model.predict(X_test).round(2)[0]
    return f"Predicted apartment price: ${prediction}"


st.title("Mexico City Housing Price Predictor 🇲🇽")

# Creating the inputs (equivalent to your Slider/Dropdowns)
area = st.slider(
    "Area (m²)",
    min_value=int(X_train["surface_covered_in_m2"].min()),
    max_value=int(X_train["surface_covered_in_m2"].max()),
    value=int(X_train["surface_covered_in_m2"].mean())
)

lat = st.slider(
    "Latitude",
    min_value=float(X_train["lat"].min()),
    max_value=float(X_train["lat"].max()),
    value=float(X_train["lat"].mean()),
    step=0.01
)

lon = st.slider(
    "Longitude",
    min_value=float(X_train["lon"].min()),
    max_value=float(X_train["lon"].max()),
    value=float(X_train["lon"].mean()),
    step=0.01
)

borough = st.selectbox(
    "Borough",
    options=sorted(X_train["borough"].unique())
)

# Display the prediction result
result = make_prediction(area, lat, lon, borough)
st.header(result)



