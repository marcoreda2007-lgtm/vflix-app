import streamlit as st
import pandas as pd
import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity

# 1. SETTING HALAMAN & STYLE PREMIUM
st.set_page_config(page_title="vflix-app", layout="wide", page_icon="🍿")

hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #4a0404 0%, #2b0202 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #4a0404 0%, #2b0202 100%);
}

/* Semua teks sidebar */
[data-testid="stSidebar"] * {
    color: #f5f5f5;
}

/* Input, selectbox, textarea di sidebar */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #0b1020 !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
}

/* Kotak info di sidebar */
[data-testid="stSidebar"] .stAlert {
    background-color: rgba(255,255,255,0.08) !important;
    color: white !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* Tombol umum */
.stButton button {
    border-radius: 8px;
    background: linear-gradient(90deg, #8b0000, #b91c1c);
    color: white;
    border: none;
}

/* Text area */
.stTextArea textarea {
    border-radius: 10px;
}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# ==========================================
# 2. LOAD DATA DAN MODEL (SYSTEM BACKEND)
# ==========================================
@st.cache_data
def load_movie_data():
    return pd.read_csv('data/movies_scored_final.csv', engine='python', on_bad_lines='skip')


@st.cache_data
def load_review_data():
    return pd.read_csv('data/reviews_final.csv', engine='python', on_bad_lines='skip')

df_movies = load_movie_data()
df_reviews = load_review_data()  # Jangan lupa inisialisasi variabelnya

@st.cache_resource
def load_ai_models():
    tfidf_vec = joblib.load('models/tfidf_rekomendasi.pkl')
    tfidf_mat = joblib.load('models/tfidf_matrix.pkl')
    return tfidf_vec, tfidf_mat


# ==========================================
# FUNGSI TRAILER (TMDB API)
# ==========================================
TMDB_API_KEY = "acf085605ee44ecca3febf0323d40329"

@st.cache_data(ttl=3600)
def get_trailer_key(movie_id):
    """
    Fetch YouTube trailer key dari TMDB.
    Return: string YouTube video key, atau None kalau tidak ada.
    """
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
        params = {"api_key": TMDB_API_KEY, "language": "en-US"}
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
        for video in data.get("results", []):
            if video.get("type") == "Trailer" and video.get("site") == "YouTube":
                return video["key"]
    except Exception:
        pass
    return None


def render_trailer_section(movie_id, unique_key):
    """
    Tampilkan tombol trailer. Kalau diklik → embed YouTube muncul di bawahnya.
    unique_key: string unik per film supaya session_state tidak bentrok.
    """
    state_key = f"show_trailer_{unique_key}"

    # Inisialisasi state
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    # Tombol toggle
    btn_label = "⏹️ Tutup Trailer" if st.session_state[state_key] else "▶️ Tonton Trailer"
    if st.button(btn_label, key=f"btn_{unique_key}"):
        st.session_state[state_key] = not st.session_state[state_key]

    # Embed trailer kalau state aktif
    if st.session_state[state_key]:
        trailer_key = get_trailer_key(movie_id)
        if trailer_key:
            embed_html = f"""
            <div style="margin-top:10px; border-radius:12px; overflow:hidden;">
                <iframe
                    width="100%"
                    height="315"
                    src="https://www.youtube.com/embed/{trailer_key}?autoplay=1&rel=0"
                    frameborder="0"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowfullscreen>
                </iframe>
            </div>
            """
            st.markdown(embed_html, unsafe_allow_html=True)
        else:
            st.warning("Trailer tidak tersedia untuk film ini.")


# Inisialisasi data dan model
df_movies = load_movie_data()
df_reviews = load_review_data()
tfidf_vec, tfidf_mat = load_ai_models()


# ==========================================
# 3. HEADER UTAMA APLIKASI
# ==========================================
st.title("🍿 VFLIX-APP")
c1, c2, c3 = st.columns(3)

c1.metric("🎬 Total Movies", "3752")
c2.metric("⭐ Avg Sentiment", "86%")
c3.metric("🧠 AI Powered", "TF-IDF + NLP")
st.markdown("### Platform Rekomendasi & Penjelajah Film Berbasis Model Sentiment Analysis")

with st.container():
    st.markdown("#### 👨‍💻 Tim Pengembang")
    col1, col2 = st.columns([2, 3])

    with col1:
        st.image("assets/image.png", use_container_width=True)

    with col2:
        st.markdown("""
        Tim:
        - Hanif
        - Reda
        - Nabilah
        """)
st.divider()


# ==========================================
# 4. IMPLEMENTASI TABS UTAMA
# ==========================================
tab_katalog, tab_ai, tab_insight = st.tabs(["🔎 Explore", "⚙️ Search with AI", "📊 Data Insight"])
# ------------------------------------------
# TAB 1: KATALOG FILM
# ------------------------------------------
with tab_katalog:
    st.subheader("Movie List ")

    with st.sidebar:
        st.header("Search & Filter")
        search_query = st.text_input(
            "Cari Judul Film:", placeholder="Ketik judul di sini...")

        genre_lists = df_movies['genres'].dropna().str.split(',')
        unique_genres = set()
        for sublist in genre_lists:
            for g in sublist:
                unique_genres.add(g.strip())

        genres_options = ["Semua Genre"] + sorted(list(unique_genres))
        selected_genre = st.selectbox("Pilih Genre:", genres_options)

        sort_option = st.selectbox("Urutkan Berdasarkan:", [
                                   "Rating Tertinggi", "Rating Terburuk"])
        st.divider()

    filtered_df = df_movies.dropna(subset=['avg_predicted_sentiment'])

    if search_query:
        filtered_df = filtered_df[filtered_df['title'].str.contains(
            search_query, case=False, na=False)]

    if selected_genre != "Semua Genre":
        filtered_df = filtered_df[filtered_df['genres'].str.contains(
            selected_genre, na=False)]

    if sort_option == "Rating Tertinggi":
        recommended_df = filtered_df.sort_values(
            by=['avg_predicted_sentiment', 'num_reviews_analyzed'], ascending=[False, False])
    else:
        recommended_df = filtered_df.sort_values(
            by=['avg_predicted_sentiment', 'num_reviews_analyzed'], ascending=[True, False])

    items_per_page = 15
    total_films = len(recommended_df)
    total_pages = max(1, (total_films + items_per_page - 1) // items_per_page)

    with st.sidebar:
        st.write(f"Data Berhasil Ditemukan: **{total_films}** judul")
        current_page = st.number_input(
            f"Halaman (1 - {total_pages}):", min_value=1, max_value=total_pages, value=1, step=1)
        st.info("💡 Sistem otomatis membagi 15 data per halaman agar performa render aplikasi tetap ringan dan ngebut!")

    start_idx = (current_page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    page_df = recommended_df.iloc[start_idx:end_idx]

    if page_df.empty:
        st.warning(
            "Film yang kamu cari tidak ditemukan. Coba ubah kata kunci pencarian atau filter genrenya ya!")
    else:
        st.write(
            f"Menampilkan urutan film halaman **{current_page}** dari total **{total_pages}**")
        st.divider()

        for index, row in page_df.iterrows():
            col_poster, col_detail = st.columns([1, 4])

            with col_poster:
                if pd.notna(row['poster_path']):
                    st.image(
                        f"https://image.tmdb.org/t/p/w500{row['poster_path']}", use_container_width=True)
                else:
                    st.image(
                        "https://via.placeholder.com/500x750?text=No+Poster", use_container_width=True)

            with col_detail:
                st.subheader(f"{row['title']}")

                if pd.notna(row['genres']):
                    clean_genres = row['genres'].replace(',', '  |  ')
                    st.markdown(f"🏷️ *{clean_genres}*")

                score_pct = int(row['avg_predicted_sentiment'] * 100)

                if score_pct >= 90:
                    status_label = "🔥 Wajib nonton! (Sangat Positif)"
                elif score_pct >= 75:
                    status_label = "🍿 Recommended (Banyak Respon Bagus)"
                elif score_pct >= 50:
                    status_label = "🤔 Lumayan (Sentimen Campuran)"
                else:
                    status_label = "📉 Kurang Bagus (Sentimen Negatif)"

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Community Score", f"{score_pct}%")
                with m2:
                    st.metric("Volume Reviews",
                              f"{int(row['num_reviews_analyzed'])} Reviews")
                with m3:
                    st.write(f"**Status Vibe:** \n\n {status_label}")

                st.progress(row['avg_predicted_sentiment'])
                st.markdown("**Sinopsis Alur Cerita:**")
                st.write(row['overview'])

                # === TRAILER SECTION ===
                st.markdown("**🎬 Trailer Film:**")
                render_trailer_section(row['id'], unique_key=f"katalog_{index}")

                # === HIGHLIGHT REVIEW ===
                st.markdown("<br>**💬 Highlight Ulasan Penonton:**",
                            unsafe_allow_html=True)

                movie_reviews = df_reviews[df_reviews['movie_id'] == row['id']]

                if not movie_reviews.empty:
                    highlight_review = movie_reviews.iloc[0]

                    if highlight_review['predicted_sentiment'] == 1:
                        sentimen_teks = "Very Positif 🔥"
                    else:
                        sentimen_teks = "Mixed feelings 🤔"

                    st.info(
                        f"*{highlight_review['review_text']}* \n\n**AI Sentimen:** {sentimen_teks}")

                else:
                    st.caption("Belum ada data sentimen review untuk divisualisasikan.")

                st.divider()

with tab_insight:
    st.subheader("Data Insight")

    genre_count = (
        df_movies['genres'].dropna()
        .str.split(',')
        .explode()
        .str.strip()
        .value_counts()
        .head(10)
    )

    st.bar_chart(genre_count)


# ------------------------------------------
# TAB 2: AI RECOMMENDER
# ------------------------------------------
with tab_ai:
    st.subheader("Tuliskan Vibe Film yang ingin anda tonton")
    st.write("Ketik plot, suasana, atau review model film yang kamu ingin. AI bakal langsung mencari film yang kamu mau*!")

    user_text = st.text_area(
        "Ceritakan jalan cerita atau suasana filmnya di sini:",
        placeholder="Contoh: Pengen nyari film detektif misteri teka-teki pembunuhan di dalam rumah tua yang gelap dan menegangkan...",
        height=120
    )

    user_rating = st.slider(
        "Set target ekspektasi rating cerita lo (1 = Biasa Saja, 5 = Masterpiece):", 1, 5, 5)

    if st.button("Analisis Teks & Cari Rekomendasi AI", type="primary"):
        if user_text:
            user_vec = tfidf_vec.transform([user_text])
            sim_scores = cosine_similarity(user_vec, tfidf_mat).flatten()
            top_indices = sim_scores.argsort()[-5:][::-1]

            st.success("Ini 5 rekomendasi teratas yang paling sesuai sama yang kamu cari:")
            st.divider()

            for idx in top_indices:
                row = df_movies.iloc[idx]
                kecocokan = int(sim_scores[idx] * 100)

                if user_rating >= 4:
                    label_rating = "🌟 Plot cerita berbobot tinggi, sangat pas dengan seleramu!"
                else:
                    label_rating = "👀 Alur cerita ringan, cocok buat hiburan santai tanpa mikir keras."

                col_poster, col_detail = st.columns([1, 4])

                with col_poster:
                    if pd.notna(row['poster_path']):
                        st.image(
                            f"https://image.tmdb.org/t/p/w500{row['poster_path']}", use_container_width=True)
                    else:
                        st.image(
                            "https://via.placeholder.com/500x750?text=No+Poster", use_container_width=True)

                with col_detail:
                    st.subheader(f"🎬 {row['title']}")

                    if pd.notna(row['genres']):
                        clean_genres = row['genres'].replace(',', '  |  ')
                        st.markdown(f"🏷️ *{clean_genres}*")

                    st.write(f"**Analisis Relevansi:** {label_rating}")
                    st.metric("Tingkat Kemiripan Cerita (AI Match)", f"{kecocokan}%")
                    st.progress(sim_scores[idx])

                    st.markdown("**Sinopsis Singkat:**")
                    st.write(row['overview'])

                    # === TRAILER SECTION ===
                    st.markdown("**🎬 Trailer Film:**")
                    render_trailer_section(row['id'], unique_key=f"ai_{idx}")

                st.divider()
        else:
            st.warning(
                "Kolom ini tidak bisa kosong ya! Coba tulis sedikit cerita atau suasana film yang kamu inginkan, biar AI bisa bantu cariin rekomendasi yang pas buat kamu!")