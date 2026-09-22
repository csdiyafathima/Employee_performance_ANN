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

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0,
    step=1.0
)

training_hours = st.number_input(
    "Training Hours",
    min_value=0.0,
    max_value=100.0,
    value=20.0,
    step=1.0
)

if st.button("Predict Performance"):

    input_data = np.array(
        [[attendance, training_hours]],
        dtype=np.float32
    )

    prediction = model.predict(input_data, verbose=0)

    probability = float(prediction[0][0])

    if probability >= 0.70:
        st.success("✅ High Performance")
        st.write(f"Performance Probability: {probability * 100:.2f}%")

    elif probability >= 0.40:
        st.warning("⚠️ Needs Improvement")
        st.write(f"Performance Probability: {probability * 100:.2f}%")

    else:
        st.error("❌ Low Performance")
        st.write(f"Performance Probability: {probability * 100:.2f}%")

    st.write("### Employee Details")
    st.write(f"**Attendance:** {attendance:.0f}%")
    st.write(f"**Training Hours:** {training_hours:.0f}")
