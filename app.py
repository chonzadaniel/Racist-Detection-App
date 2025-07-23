# app.py
import streamlit as st
from inference import predict

st.set_page_config(page_title="Racist Tweet Detector", layout="centered")

st.title("📝 Racist Tweet Detector")
st.write("Enter a tweet below and let the model predict whether it is *racist* or *not*.")

tweet = st.text_area("Tweet Text", height=80)

if st.button("Predict"):
    if tweet.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing…"):
            label_id, label_name, confidence = predict(tweet)
            st.success(f"Prediction: **{label_name}**")
            st.info(f"📊 Confidence: {confidence:.2%}")
