import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="ER Wait Time Simulator", layout="wide")

st.title("ER Wait Time Simulator")
st.write(
    "Adjust patient and hospital variables to simulate predicted emergency room wait times."
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("ER_Wait_Time_Dataset.csv")
    return df

df = load_data()

# -----------------------------
# PREPROCESSING
# Matches your notebook backend
# -----------------------------
@st.cache_data
def preprocess_data(df):
    df = df.copy()

    # Encode ordinal features
    urgency_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    df['Urgency Encoded'] = df['Urgency Level'].map(urgency_map)

    time_map = {'Morning': 1, 'Late Morning': 2, 'Afternoon': 3, 'Evening': 4, 'Night': 5}
    df['Time Encoded'] = df['Time of Day'].map(time_map)

    # One-hot encode Season
    season_dummies = pd.get_dummies(df['Season'], prefix='Season', drop_first=True)
    df = pd.concat([df, season_dummies], axis=1)
    season_features = season_dummies.columns.tolist()

    # Encode Region
    region_encoder = LabelEncoder()
    df['Region Encoded'] = region_encoder.fit_transform(df['Region'])

    # Weekend feature
    df['Is Weekend'] = df['Day of Week'].isin(['Saturday', 'Sunday']).astype(int)

    # Final feature set from your notebook
    features = [
        'Urgency Encoded',
        'Nurse-to-Patient Ratio',
        'Time Encoded',
        'Region Encoded',
        'Is Weekend',
    ] + season_features

    df_clean = df[features + ['Total Wait Time (min)']].dropna()

    X = df_clean[features]
    y = df_clean['Total Wait Time (min)']

    return df_clean, X, y, features, region_encoder, season_features

df_clean, X, y, features, region_encoder, season_features = preprocess_data(df)

# -----------------------------
# TRAIN MODELS
# -----------------------------
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    return lr_model, rf_model, gb_model

lr_model, rf_model, gb_model = train_models(X, y)

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Simulation Inputs")

selected_model = st.sidebar.selectbox(
    "Choose Prediction Model",
    ["Linear Regression", "Random Forest", "Gradient Boosting"]
)

urgency = st.sidebar.selectbox("Urgency Level", ['Low', 'Medium', 'High', 'Critical'])

time_of_day = st.sidebar.selectbox(
    "Time of Day",
    ['Morning', 'Late Morning', 'Afternoon', 'Evening', 'Night']
)

day_of_week = st.sidebar.selectbox(
    "Day of Week",
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)

season = st.sidebar.selectbox(
    "Season",
    sorted(df['Season'].dropna().unique().tolist())
)

region = st.sidebar.selectbox(
    "Region",
    sorted(df['Region'].dropna().unique().tolist())
)

nurse_ratio = st.sidebar.slider(
    "Nurse-to-Patient Ratio",
    min_value=float(df['Nurse-to-Patient Ratio'].min()),
    max_value=float(df['Nurse-to-Patient Ratio'].max()),
    value=float(df['Nurse-to-Patient Ratio'].median()),
    step=0.1
)

# -----------------------------
# BUILD USER INPUT ROW
# -----------------------------
def build_input_row(
    urgency, time_of_day, day_of_week, season, region, nurse_ratio,
    region_encoder, season_features
):
    urgency_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
    time_map = {'Morning': 1, 'Late Morning': 2, 'Afternoon': 3, 'Evening': 4, 'Night': 5}

    is_weekend = 1 if day_of_week in ['Saturday', 'Sunday'] else 0
    region_encoded = region_encoder.transform([region])[0]

    row = {
        'Urgency Encoded': urgency_map[urgency],
        'Nurse-to-Patient Ratio': nurse_ratio,
        'Time Encoded': time_map[time_of_day],
        'Region Encoded': region_encoded,
        'Is Weekend': is_weekend
    }

    # Add season dummy columns
    for col in season_features:
        row[col] = 0

    season_col = f"Season_{season}"
    if season_col in row:
        row[season_col] = 1

    return pd.DataFrame([row])

input_df = build_input_row(
    urgency, time_of_day, day_of_week, season, region, nurse_ratio,
    region_encoder, season_features
)

# Ensure same column order
input_df = input_df[features]

# -----------------------------
# SELECT MODEL
# -----------------------------
if selected_model == "Linear Regression":
    model = lr_model
elif selected_model == "Random Forest":
    model = rf_model
else:
    model = gb_model

prediction = model.predict(input_df)[0]

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
st.subheader("Predicted Wait Time")
st.metric("Estimated Total Wait Time", f"{prediction:.2f} minutes")

if prediction < 60:
    st.success("Low predicted wait time")
elif prediction < 120:
    st.warning("Moderate predicted wait time")
else:
    st.error("High predicted wait time")

# -----------------------------
# SCENARIO COMPARISON
# -----------------------------
st.subheader("Scenario Comparison")

improved_input = input_df.copy()
improved_input['Nurse-to-Patient Ratio'] = min(
    improved_input['Nurse-to-Patient Ratio'].iloc[0] + 1.0,
    df['Nurse-to-Patient Ratio'].max()
)

improved_prediction = model.predict(improved_input)[0]
difference = prediction - improved_prediction

col1, col2 = st.columns(2)

with col1:
    st.write("**Current Scenario**")
    st.dataframe(input_df)

with col2:
    st.write("**Improved Staffing Scenario**")
    st.dataframe(improved_input)

st.metric(
    "Predicted Reduction if Nurse-to-Patient Ratio Increases by 1.0",
    f"{difference:.2f} minutes"
)

# -----------------------------
# FEATURE EFFECT FOR LINEAR REGRESSION
# -----------------------------
if selected_model == "Linear Regression":
    st.subheader("Model Coefficients")

    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": lr_model.coef_
    }).sort_values("Coefficient", key=abs, ascending=False)

    st.dataframe(coef_df, use_container_width=True)

    st.write(
        "For linear regression, larger absolute coefficients indicate stronger influence on predicted wait time."
    )