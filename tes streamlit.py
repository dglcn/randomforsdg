import streamlit as st
import joblib
import re

# Load Model dan Vectorizer
model = joblib.load("randomfor.pkl")
# vectorizer = joblib.load("vectorizer.pkl")

# Fungsi Prediksi
def predict_sdg(title, abstract):
    text = title + " " + abstract
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]  # Ambil hasil prediksi pertama
    sdg_labels = ["SDG1", "SDG2", "SDG3", ..., "SDG16"]  # Sesuaikan dengan label SDG kamu
    matched_sdgs = [sdg_labels[i] for i, val in enumerate(prediction) if val == 1]
    return matched_sdgs if matched_sdgs else ["No Match"]

# UI dengan Streamlit
st.title("SDG Classification for Research Papers")
st.write("Masukkan **Title** dan **Abstract** untuk mendapatkan kategori SDG.")

title = st.text_area("Masukkan Title")
abstract = st.text_area("Masukkan Abstract")

if st.button("Prediksi SDG"):
    if title.strip() and abstract.strip():
        result = predict_sdg(title, abstract)
        st.success(f"SDG yang cocok: {', '.join(result)}")
    else:
        st.warning("Harap isi Title dan Abstract!")

