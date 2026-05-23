import streamlit as st
import pandas as pd
import joblib
import requests
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
import streamlit.components.v1 as components

# ==========================================
# 1. SETTING HALAMAN & STYLE PREMIUM
# ==========================================
st.set_page_config(page_title="vflix-app", layout="wide", page_icon="🍿")

hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Sidebar selalu terbuka dan sembunyikan tombol toggle */
[data-testid="stSidebar"] {
    transform: none !important;
    min-width: 244px !important;
    max-width: 244px !important;
}
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
button[data-testid="baseButton-headerNoPadding"] { display: none !important; }
[data-testid="stSidebarHeader"] button { display: none !important; }

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #4a0404 0%, #2b0202 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #4a0404 0%, #2b0202 100%);
}
[data-testid="stSidebar"] * { color: #f5f5f5; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #0b1020 !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
}
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
.stTextArea textarea { border-radius: 10px; }
</style>


"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# ==========================================
# 2. LOAD DATA DAN MODEL
# ==========================================
@st.cache_data
def load_movie_data():
    return pd.read_csv('data/movies_scored_final.csv', engine='python', on_bad_lines='skip')

@st.cache_data
def load_review_data():
    return pd.read_csv('data/reviews_final.csv', engine='python', on_bad_lines='skip')

@st.cache_resource
def load_ai_models():
    tfidf_vec = joblib.load('models/tfidf_rekomendasi.pkl')
    tfidf_mat = joblib.load('models/tfidf_matrix.pkl')
    return tfidf_vec, tfidf_mat

# ==========================================
# FUNGSI TRAILER (TMDB API)
# ==========================================
TMDB_API_KEY = "acf085605ee44ecca3febf0323d40329"
# TMDB_API_KEY = os.getenv("TMDB_API_KEY")

@st.cache_data(ttl=3600)
def get_trailer_key(movie_id):
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
    state_key = f"show_trailer_{unique_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    btn_label = "⏹️ Tutup Trailer" if st.session_state[state_key] else "▶️ Tonton Trailer"
    if st.button(btn_label, key=f"btn_{unique_key}"):
        st.session_state[state_key] = not st.session_state[state_key]

    if st.session_state[state_key]:
        trailer_key = get_trailer_key(movie_id)
        if trailer_key:
            embed_html = f"""
            <div style="margin-top:10px; border-radius:12px; overflow:hidden;">
                <iframe width="100%" height="315"
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

# ==========================================
# INISIALISASI
# ==========================================
df_movies = load_movie_data()
df_reviews = load_review_data()
tfidf_vec, tfidf_mat = load_ai_models()
load_dotenv()

# ==========================================
# 3. HEADER UTAMA
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
# SIDEBAR — di luar tab agar selalu tampil
# ==========================================
with st.sidebar:
    st.header("Search & Filter")
    search_query = st.text_input("Cari Judul Film:", placeholder="Ketik judul di sini...")

    genre_lists = df_movies['genres'].dropna().str.split(',')
    unique_genres = set()
    for sublist in genre_lists:
        for g in sublist:
            unique_genres.add(g.strip())

    genres_options = ["Semua Genre"] + sorted(list(unique_genres))
    selected_genre = st.selectbox("Pilih Genre:", genres_options)
    sort_option = st.selectbox("Urutkan Berdasarkan:", ["Rating Tertinggi", "Rating Terburuk"])
    st.divider()

# Hitung filter & pagination sebelum tab dimulai
filtered_df = df_movies.dropna(subset=['avg_predicted_sentiment'])

if search_query:
    filtered_df = filtered_df[filtered_df['title'].str.contains(search_query, case=False, na=False)]

if selected_genre != "Semua Genre":
    filtered_df = filtered_df[filtered_df['genres'].str.contains(selected_genre, na=False)]

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

# ==========================================
# 4. TABS UTAMA
# ==========================================
tab_katalog, tab_ai, tab_insight, tab_tren = st.tabs(["🔎 Explore", "⚙️ Search with AI", "📊 Data Insight", "🏆 Tren Film"])

# ------------------------------------------
# TAB 1: KATALOG FILM
# ------------------------------------------
with tab_katalog:
    st.subheader("Movie List")

    start_idx = (current_page - 1) * items_per_page
    end_idx   = start_idx + items_per_page
    page_df   = recommended_df.iloc[start_idx:end_idx]

    if page_df.empty:
        st.warning("Film yang kamu cari tidak ditemukan. Coba ubah kata kunci pencarian atau filter genrenya ya!")
    else:
        st.write(f"Menampilkan urutan film halaman **{current_page}** dari total **{total_pages}**")
        st.divider()

        for index, row in page_df.iterrows():
            col_poster, col_detail = st.columns([1, 4])

            with col_poster:
                if pd.notna(row['poster_path']):
                    st.image(f"https://image.tmdb.org/t/p/w500{row['poster_path']}", use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/500x750?text=No+Poster", use_container_width=True)

            with col_detail:
                st.subheader(f"{row['title']}")
                if pd.notna(row['genres']):
                    st.markdown(f"🏷️ *{row['genres'].replace(',', '  |  ')}*")

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
                    st.metric("Volume Reviews", f"{int(row['num_reviews_analyzed'])} Reviews")
                with m3:
                    st.write(f"**Status Vibe:** \n\n {status_label}")

                st.progress(row['avg_predicted_sentiment'])
                st.markdown("**Sinopsis Alur Cerita:**")
                st.write(row['overview'])
                st.markdown("**🎬 Trailer Film:**")
                render_trailer_section(row['id'], unique_key=f"katalog_{index}")

                st.markdown("<br>**💬 Highlight Ulasan Penonton:**", unsafe_allow_html=True)
                movie_reviews = df_reviews[df_reviews['movie_id'] == row['id']]
                if not movie_reviews.empty:
                    highlight_review = movie_reviews.iloc[0]
                    sentimen_teks = "Very Positif 🔥" if highlight_review['predicted_sentiment'] == 1 else "Mixed feelings 🤔"
                    st.info(f"*{highlight_review['review_text']}* \n\n**AI Sentimen:** {sentimen_teks}")
                else:
                    st.caption("Belum ada data sentimen review untuk divisualisasikan.")

                st.divider()

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
    user_rating = st.slider("Set target ekspektasi rating cerita lo (1 = Biasa Saja, 5 = Masterpiece):", 1, 5, 5)

    if st.button("Analisis Teks & Cari Rekomendasi AI", type="primary", key="btn_rekom_ai"):
        if user_text:
            user_vec   = tfidf_vec.transform([user_text])
            sim_scores = cosine_similarity(user_vec, tfidf_mat).flatten()
            top_indices = sim_scores.argsort()[-5:][::-1]
            st.session_state["ai_top_indices"] = top_indices
            st.session_state["ai_sim_scores"]  = sim_scores
            st.session_state["ai_user_rating"] = user_rating
        else:
            st.warning("Kolom ini tidak bisa kosong ya!")

    if "ai_top_indices" in st.session_state:
        top_indices = st.session_state["ai_top_indices"]
        sim_scores  = st.session_state["ai_sim_scores"]
        user_rating = st.session_state["ai_user_rating"]

        st.success("Ini 5 rekomendasi teratas yang paling sesuai sama yang kamu cari:")
        st.divider()

        for idx in top_indices:
            row = df_movies.iloc[idx]
            kecocokan = int(sim_scores[idx] * 100)
            label_rating = (
                "🌟 Plot cerita berbobot tinggi, sangat pas dengan seleramu!"
                if user_rating >= 4
                else "👀 Alur cerita ringan, cocok buat hiburan santai tanpa mikir keras."
            )

            col_poster, col_detail = st.columns([1, 4])
            with col_poster:
                if pd.notna(row['poster_path']):
                    st.image(f"https://image.tmdb.org/t/p/w500{row['poster_path']}", use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/500x750?text=No+Poster", use_container_width=True)

            with col_detail:
                st.subheader(f"🎬 {row['title']}")
                if pd.notna(row['genres']):
                    st.markdown(f"🏷️ *{row['genres'].replace(',', '  |  ')}*")
                st.write(f"**Analisis Relevansi:** {label_rating}")
                st.metric("Tingkat Kemiripan Cerita (AI Match)", f"{kecocokan}%")
                st.progress(sim_scores[idx])
                st.markdown("**Sinopsis Singkat:**")
                st.write(row['overview'])
                st.markdown("**🎬 Trailer Film:**")
                render_trailer_section(row['id'], unique_key=f"ai_trailer_{idx}")

            st.divider()

# ------------------------------------------
# TAB 3: DATA INSIGHT
# ------------------------------------------
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
# TAB 4: TREN FILM
# ------------------------------------------
with tab_tren:

    if 'year' not in df_movies.columns:
        df_movies['year'] = pd.to_datetime(
            df_movies.get('release_date', pd.Series(dtype=str)),
            errors='coerce'
        ).dt.year

    available_years = sorted(df_movies['year'].dropna().astype(int).unique().tolist())

    if 'tren_year' not in st.session_state:
        st.session_state['tren_year'] = (
            2024 if 2024 in available_years
            else (available_years[-1] if available_years else 2024)
        )

    current_year = int(st.session_state['tren_year'])

    BASE_IMG  = "https://image.tmdb.org/t/p/w500"
    NO_POSTER = "https://via.placeholder.com/500x750?text=No+Poster"

    def poster_url(row):
        return f"{BASE_IMG}{row['poster_path']}" if pd.notna(row.get('poster_path')) else NO_POSTER

    year_df = df_movies[df_movies['year'] == current_year].dropna(subset=['avg_predicted_sentiment'])
    top5 = (
        year_df
        .sort_values(by=['avg_predicted_sentiment', 'num_reviews_analyzed'], ascending=[False, False])
        .head(5)
        .reset_index(drop=True)
    )

    display_order = [3, 1, 0, 2, 4]
    rank_labels   = ["TOP 4", "TOP 2", "TOP 1", "TOP 3", "TOP 5"]
    img_heights   = ["140px", "190px", "255px", "190px", "140px"]

    cards_inner = ""
    for slot_idx, (film_idx, rank, h) in enumerate(zip(display_order, rank_labels, img_heights)):
        is_center = (slot_idx == 2)

        if film_idx < len(top5):
            row        = top5.iloc[film_idx]
            img        = poster_url(row)
            title      = row['title']
            score      = int(row['avg_predicted_sentiment'] * 100)
            short_title = (title[:18] + "…") if len(title) > 18 else title

            rank_color = "#ff4444" if is_center else "#cc2222"
            title_size = "13px"   if is_center else "11px"
            title_fw   = "700"    if is_center else "500"
            rank_size  = "15px"   if is_center else "11px"
            shadow     = "0 8px 28px rgba(0,0,0,0.7)" if is_center else "0 4px 14px rgba(0,0,0,0.45)"

            cards_inner += f"""
            <div style="display:flex;flex-direction:column;align-items:center;flex:0 0 auto;">
                <img src="{img}" style="height:{h};border-radius:10px;object-fit:cover;box-shadow:{shadow};display:block;">
                <div style="margin-top:9px;text-align:center;max-width:130px;">
                    <div style="font-size:{rank_size};font-weight:700;color:{rank_color};letter-spacing:0.5px;">{rank}</div>
                    <div style="font-size:{title_size};font-weight:{title_fw};color:#f0f0f0;line-height:1.35;margin-top:3px;">{short_title}</div>
                    <div style="font-size:10px;color:#aaaaaa;margin-top:3px;">⭐ {score}%</div>
                </div>
            </div>
            """
        else:
            cards_inner += f'<div style="flex:0 0 auto;width:80px;height:{h};"></div>'

    full_html = f"""
    <div style="
        border:2.5px solid #b91c1c;
        border-radius:18px;
        padding:26px 24px 28px 24px;
        background:#0f1117;
        font-family:'Segoe UI',sans-serif;
        box-sizing:border-box;
    ">
        <div style="font-size:20px;font-weight:700;color:#f0f0f0;margin-bottom:22px;">
            🔥 Tren Film Teratas
        </div>
        <div style="display:flex;align-items:flex-end;justify-content:center;gap:18px;padding-bottom:6px;">
            {cards_inner}
        </div>
    </div>
    """

    components.html(full_html, height=460, scrolling=False)

    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

    prev_year = current_year - 1
    next_year = current_year + 1

    col_prev, col_cur, col_next = st.columns([1, 2, 1])

    with col_prev:
        if prev_year in available_years:
            if st.button(f"◀  {prev_year}", key="tren_prev", use_container_width=True):
                st.session_state['tren_year'] = prev_year
                st.rerun()

    with col_cur:
        st.markdown(f"""
            <div style="
                text-align:center;font-size:28px;font-weight:800;
                border:2.5px solid #b91c1c;border-radius:12px;
                padding:6px 0 7px 0;color:#f0f0f0;
                letter-spacing:2px;line-height:1.3;
            ">{current_year}</div>
        """, unsafe_allow_html=True)

    with col_next:
        if next_year in available_years:
            if st.button(f"{next_year}  ▶", key="tren_next", use_container_width=True):
                st.session_state['tren_year'] = next_year
                st.rerun()

    st.markdown(f"<br>📋 Detail Top 5 Film Tahun {current_year}", unsafe_allow_html=True)

    if top5.empty:
        st.warning(f"Tidak ada data film untuk tahun {current_year}.")
    else:
        for idx, row in top5.iterrows():
            score_pct = int(row['avg_predicted_sentiment'] * 100)
            with st.expander(f"#{idx + 1}  —  {row['title']}  |  ⭐ {score_pct}%"):
                col_p, col_d = st.columns([1, 3])
                with col_p:
                    st.image(poster_url(row), use_container_width=True)
                with col_d:
                    if pd.notna(row.get('genres')):
                        st.markdown(f"🏷️ *{row['genres'].replace(',', '  |  ')}*")
                    st.metric("Community Score", f"{score_pct}%")
                    st.progress(row['avg_predicted_sentiment'])
                    st.write(row.get('overview', '-'))
                    render_trailer_section(row['id'], unique_key=f"tren_{current_year}_{idx}")