import streamlit as st
from evaluate import predict

st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Prediction System")
st.write("Enter any news text and check whether it is Fake or Real")

input_data = st.text_area("Enter News Content")

if st.button("Predict"):
    if input_data.strip() == "":
        st.warning("Please enter news text")
    else:
        pred = predict(input_data)

        # confidence
        confidence = float(pred)

        st.write("### Result")

        if confidence > 0.5:
            st.success("🟢 Real News")
        else:
            st.error("🔴 Fake News")

        st.write(f"**Confidence Score:** {confidence:.4f}")

        # confidence bar
        st.progress(min(confidence, 1.0))