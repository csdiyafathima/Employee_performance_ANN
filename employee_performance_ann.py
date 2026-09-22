
import streamlit as st
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("employee_performance_ann.keras")

st.set_page_config(
    page_title="Employee Performance Prediction",
    page_icon="👨‍💼",
    layout="centered"
)

st.title(" Employee Performance Prediction")
st.write("Enter the employee details to predict performance.")

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=40.0,
    value=2.0,
    step=0.5
)

hours_worked = st.number_input(
    "Hours Worked per Week",
    min_value=0.0,
    max_value=100.0,
    value=40.0,
    step=1.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=90.0,
    step=1.0
)

if st.button("Predict Performance"):

    input_data = np.array([
        [experience, hours_worked, attendance]
    ])

    prediction = model.predict(input_data)

    st.write("### Prediction")

    if prediction[0][0] >= 0.5:
        st.success("✅ High Performance")
    else:
        st.warning("⚠️ Low Performance")

    probability = prediction[0][0] * 100

    st.write(f"Prediction Probability: {probability:.2f}%")

