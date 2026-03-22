import joblib
import pandas as pd
import streamlit as st

# Load the trained model
model = joblib.load("logistic_regression_pipline.pkl")
st.title("Personality Type Prediction App")
st.markdown("Please Enter The Text To Predict The Personality Type")
text_input = st.text_area("Enter Text Here")
if st.button("Predict"):
    input_data = pd.DataFrame(
        [
            {
                "cleaned_posts": text_input,
                "words_count": len(text_input.split()),
                "text_length": len(text_input),
            }
        ]
    )
    prediction = model.predict(input_data)
    st.subheader(f"Prediction Result: {prediction[0]}")
    st.markdown(
        """
        KEYs:
        
                Introversion (I) \n
                Extroversion (E)\n
                Intuition (N) \n
                Sensing (S)\n
                Thinking (T) \n
                Feeling (F)\n
                Judging (J) \n
                Perceiving (P)\n
                """
    )
