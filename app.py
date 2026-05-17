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
# st.write("Daftar Kolom di CSV:", df_movies.columns.tolist())

model, vectorizer = load_model()

# Header Web
st.title("🍿 vflix: Rekomendasi Film Cerdas")
st.write("Temukan tontonan seru berdasarkan sentimen asli ulasan penonton!")

# Dropdown Filter
# 1. Ekstrak semua genre unik dari dataset secara otomatis
if 'genres' in df_movies.columns:
    # Ambil kolom genres, buang yang kosong, lalu split berdasarkan koma
    genre_lists = df_movies['genres'].dropna().str.split(',')

    # Pakai 'set' biar nama genre yang duplikat otomatis ke-filter
    unique_genres = set()
    for sublist in genre_lists:
        for g in sublist:
            unique_genres.add(g.strip())  # .strip() buat ilangin spasi gaib

    # Urutkan sesuai abjad A-Z biar rapi pas dilihat user
    genres_options = sorted(list(unique_genres))
else:
    # Jaga-jaga kalau kolomnya mendadak ilang
    genres_options = ["Action", "Drama", "Comedy"]

# 2. Masukin list dinamis tadi ke dropdown Streamlit
selected_genre = st.selectbox("Pilih Genre Pilihanmu:", genres_options)

# Tombol Eksekusi
if st.button("Cari Film"):
    # 1. Bersihin dulu film yang skor sentimennya kosong (NaN) biar sistem gak bingung
    # Filter pakai 'genres' dan str.contains, dilanjut dropna
    filtered_df = df_movies[df_movies['genres'].str.contains(selected_genre, na=False)].dropna(
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
            # 1. Konversi float ke persentase biar lebih mudah dipahami
            score_pct = int(row['avg_predicted_sentiment'] * 100)

            # 2. Bikin klasifikasi label & emoji penanda yang eye-catching
            if score_pct >= 90:
                status_label = "🔥 Tontonan Wajib! (Sangat Positif)"
            elif score_pct >= 75:
                status_label = "🍿 Recommended (Banyak Respon Bagus)"
            elif score_pct >= 50:
                status_label = "🤔 Lumayan (Sentimen Campuran)"
            else:
                status_label = "📉 Kurang Greget (Sentimen Negatif)"

            # 3. Cetak Judul Film dengan ukuran subheader yang pas
            st.subheader(f"🎬 {row['title']}")

            # 4. Gunakan layout kolom biar tampilan informasi sejajar rapi
            col_metric, col_progress = st.columns([1, 2])

            with col_metric:
                # Nampilin angka persentase besar ala dashboard profesional
                st.metric(label="Sentimen Positif", value=f"{score_pct}%")

            with col_progress:
                st.write(f"**Status Komunitas:** {status_label}")
                # Nampilin progress bar visual biar kelihatan bergerak indikatornya
                st.progress(row['avg_predicted_sentiment'])
                st.caption(
                    f"Dianalisis dari {int(row['num_reviews_analyzed'])} ulasan penonton asli.")

            # 5. Tampilkan sinopsis film
            st.write(f"**Sinopsis:** {row['overview']}")

            # Pembatas antar film biar gak numpuk
            st.divider()
