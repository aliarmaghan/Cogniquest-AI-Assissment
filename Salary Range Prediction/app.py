import streamlit as st
import pickle
import numpy as np

# # Load the saved model
# with open('model.pkl', 'rb') as file:
#     model = pickle.load(file)

# # Load any additional preprocessing steps if needed
# with open('scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)  # If you used a scaler for normalization

# App title
st.title("Job Salary Range Prediction App")

# App description
st.write("This app predicts the salary range for a job based on the job summary!")

# Input fields
st.header("Input Job Summary Details")
job_summary = st.text_area("Enter Job Summary", "Describe the job requirements and qualifications...")

# Preprocess input
if st.button("Predict Salary Range"):
    if job_summary.strip() == "":
        st.error("Please enter a valid job summary.")
    else:
        # Convert job summary to vector (use tokenizer/vectorizer used during training)
        try:
            with open('vectorizer.pkl', 'rb') as vec_file:
                vectorizer = pickle.load(vec_file)
            job_summary_vector = vectorizer.transform([job_summary])

            # Normalize vector if necessary
            job_summary_vector = scaler.transform(job_summary_vector.toarray()) if scaler else job_summary_vector

            # Make prediction
            prediction = model.predict(job_summary_vector)
            prediction_prob = model.predict_proba(job_summary_vector)

            # Display prediction
            st.success(f"Predicted Salary Range: {prediction[0]}")
            st.write(f"Confidence: {np.max(prediction_prob) * 100:.2f}%")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
