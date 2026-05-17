import streamlit as st
import pandas as pd
import joblib

# Bikin tampilan web jadi full width
st.set_page_config(page_title="vflix", layout="wide")

# Fungsi load data biar webnya ngebut pake cache


@st.cache_data
def load_data():
    # Pastiin nama filenya bener: movies_scored_final.csv
    return pd.read_csv('data/movies_scored_final.csv', engine='python', on_bad_lines='skip')


@st.cache_resource
def load_model():
    model = joblib.load('models/sentiment_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return model, vectorizer


df_movies = load_data()

# Tambahin baris ini buat ngintip kolom
st.write("Daftar Kolom di CSV:", df_movies.columns.tolist())

model, vectorizer = load_model()

# Header Web
st.title("🍿 vflix: Rekomendasi Film Cerdas")
st.write("Temukan tontonan seru berdasarkan sentimen asli ulasan penonton!")

# Dropdown Filter
genres = ["Action", "Drama", "Comedy", "Sci-Fi", "Romance", "Horror"]
selected_genre = st.selectbox("Pilih Genre Pilihanmu:", genres)

# Tombol Eksekusi
if st.button("Cari Film"):
    # 1. Bersihin dulu film yang skor sentimennya kosong (NaN) biar sistem gak bingung
    filtered_df = df_movies[df_movies['genre'] == selected_genre].dropna(
        subset=['avg_predicted_sentiment'])

    # 2. Sortir Ganda: Urutkan dari Sentimen tertinggi, lalu adu jumlah review terbanyak
    recommended_df = filtered_df.sort_values(
        by=['avg_predicted_sentiment', 'num_reviews_analyzed'],
        ascending=[False, False]
    ).head(5)

    if recommended_df.empty:
        st.warning("Waduh, belum ada film di genre ini nih.")
    else:
        st.success(f"Top rekomendasi {selected_genre} buat kamu!")
        for index, row in recommended_df.iterrows():
            st.subheader(row['title'])
            st.write(
                f"⭐ Skor Sentimen: {row['avg_predicted_sentiment']} (Berdasarkan {row['num_reviews_analyzed']} ulasan)")
            st.write(row['overview'])
            st.divider()
