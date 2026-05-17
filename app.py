import streamlit as st
import pandas as pd
import joblib

# Bikin tampilan web jadi full width
st.set_page_config(page_title="REDa Film", layout="wide")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

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

# --- HEADER UTAMA ---
st.title("🍿 REDa Film")
st.markdown("Rekomendasi Film Cerdas Berdasarkan Sentimen Penonton")
st.divider()

# === 1. SIDEBAR (PANEL SAMPING) ===
with st.sidebar:
    st.header("🔍 Filter & Cari")

    search_query = st.text_input(
        "Cari Judul Film:", placeholder="Ketik judul...")

    # Ekstrak genre dinamis
    genre_lists = df_movies['genres'].dropna().str.split(',')
    unique_genres = set()
    for sublist in genre_lists:
        for g in sublist:
            unique_genres.add(g.strip())

    genres_options = ["Semua Genre"] + sorted(list(unique_genres))
    selected_genre = st.selectbox("Pilih Genre:", genres_options)

    sort_option = st.selectbox("Urutkan Berdasarkan:", [
                               "Rating Tertinggi 🌟", "Rating Terburuk 🤢"])

    st.divider()  # Sebagai pembatas di sidebar
    st.info("💡 Algoritma vflix menganalisis ulasan asli penonton. Berani nonton yang ratingnya merah?")

# === 2. LOGIKA FILTERING & SORTING ===
filtered_df = df_movies.dropna(subset=['avg_predicted_sentiment'])

if search_query:
    filtered_df = filtered_df[filtered_df['title'].str.contains(
        search_query, case=False, na=False)]

if selected_genre != "Semua Genre":
    filtered_df = filtered_df[filtered_df['genres'].str.contains(
        selected_genre, na=False)]

if sort_option == "Rating Tertinggi 🌟":
    recommended_df = filtered_df.sort_values(
        by=['avg_predicted_sentiment', 'num_reviews_analyzed'],
        ascending=[False, False]
    )
else:
    recommended_df = filtered_df.sort_values(
        by=['avg_predicted_sentiment', 'num_reviews_analyzed'],
        ascending=[True, False]
    )

# === 3. FITUR BARU: LOGIKA PAGINATION (15 FILM PER HALAMAN) ===
items_per_page = 15
total_films = len(recommended_df)

# Hitung total halaman yang dibutuhkan
if total_films > 0:
    total_pages = (total_films + items_per_page - 1) // items_per_page
else:
    total_pages = 1

# Tambahkan input angka halaman di bagian bawah sidebar
with st.sidebar:
    st.write(f"Total data ditemukan: **{total_films}** film")
    current_page = st.number_input(
        f"Halaman (1 - {total_pages}):",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1
    )

# Potong data berdasarkan halaman yang aktif (Slicing data)
start_idx = (current_page - 1) * items_per_page
end_idx = start_idx + items_per_page
page_df = recommended_df.iloc[start_idx:end_idx]


# === 4. NAMPILIN HASIL (MENGGUNAKAN page_df BUKAN recommended_df) ===
if page_df.empty:
    st.warning("Waduh, film yang lo cari belum ada nih.")
else:
    st.write(f"Menampilkan halaman **{current_page}** dari **{total_pages}**")

    for index, row in page_df.iterrows():
        col_poster, col_detail = st.columns([1, 3])

        with col_poster:
            if pd.notna(row['poster_path']):
                poster_url = f"https://image.tmdb.org/t/p/w500{row['poster_path']}"
                st.image(poster_url, use_container_width=True)
            else:
                st.image(
                    "https://via.placeholder.com/500x750?text=No+Poster", use_container_width=True)

        with col_detail:
            st.subheader(f"{row['title']}")

            score_pct = int(row['avg_predicted_sentiment'] * 100)

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Skor Sentimen", f"{score_pct}%")
            with m2:
                st.metric("Total Ulasan",
                          f"{int(row['num_reviews_analyzed'])} Review")

            st.progress(row['avg_predicted_sentiment'])

            st.caption("SINOPSIS")
            st.write(row['overview'])

        st.divider()

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
