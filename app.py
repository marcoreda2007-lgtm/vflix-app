import streamlit as st
import pandas as pd
import joblib

# Bikin tampilan web jadi full width
st.set_page_config(page_title="vflix", layout="wide")

# Fungsi load data biar webnya ngebut pake cache


@st.cache_data
def load_data():
    # Sesuaikan dengan nama file dataset kamu
    return pd.read_csv('data/movies.csv', engine='python', on_bad_lines='skip')


@st.cache_resource
def load_model():
    model = joblib.load('models/sentiment_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return model, vectorizer


df_movies = load_data()
model, vectorizer = load_model()

# Header Web
st.title("🍿 vflix: Rekomendasi Film Cerdas")
st.write("Temukan tontonan seru berdasarkan sentimen asli ulasan penonton!")

# Dropdown Filter
genres = ["Action", "Drama", "Comedy", "Sci-Fi", "Romance", "Horror"]
selected_genre = st.selectbox("Pilih Genre Pilihanmu:", genres)

# Tombol Eksekusi
if st.button("Cari Film"):
    # Filter data berdasarkan genre yang dipilih
    filtered_df = df_movies[df_movies['genres'].str.contains(
        selected_genre, na=False)]

    # Ambil 5 film dengan sentimen tertinggi
    recommended_df = filtered_df.sort_values(
        by='average_sentiment', ascending=False).head(5)

    if recommended_df.empty:
        st.warning("Waduh, belum ada film di genre ini nih.")
    else:
        st.success(f"Top rekomendasi {selected_genre} buat kamu!")
        for index, row in recommended_df.iterrows():
            st.subheader(row['title'])
            st.write(f"⭐ Skor Sentimen: {row['average_sentiment']}")
            st.write(row['overview'])
            st.divider()
