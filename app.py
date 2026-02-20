import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

st.title("Student Performance Prediction")

df = pd.read_csv("student-mat.csv")
df = pd.get_dummies(df, drop_first=True)

X = df.drop("G3", axis=1)
y = df["G3"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

studytime = st.slider("Study Time (1-4)", 1, 4, 2)
absences = st.slider("Absences", 0, 30, 5)

input_df = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)
input_df["studytime"] = studytime
input_df["absences"] = absences

input_scaled = scaler.transform(input_df)
prediction = model.predict(input_scaled)
st.write("Predicted Final Grade:", round(prediction[0], 2))