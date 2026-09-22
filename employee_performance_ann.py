import streamlit as st
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("employee_performance_ann.keras")

st.set_page_config(
    page_title="Employee Performance Prediction",
    page_icon="👨‍💼",
    layout="centered"
)

st.title("👨‍💼 Employee Performance Prediction")
st.write("Enter the employee details to predict performance.")

training_hours = st.number_input(
    "Training Hours",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=1.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)

if st.button("Predict Performance"):

    input_data = np.array(
        [[training_hours, attendance]],
        dtype=np.float32
    )

    prediction = model.predict(input_data, verbose=0)

    probability = float(prediction[0][0])

    if probability >= 0.5:
        st.success("✅ Good Performance")
    else:
        st.warning("⚠️ Needs Improvement")

    st.write(f"Good Performance Probability: {probability * 100:.2f}%")

    st.write("### Employee Details")
    st.write(f"**Training Hours:** {training_hours:.0f}")
    st.write(f"**Attendance:** {attendance:.0f}%")
