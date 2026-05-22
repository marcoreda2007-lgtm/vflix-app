# 🍿 VFLIX-APP

![VFLIX Preview](assets/image.png)

Platform rekomendasi film berbasis AI menggunakan NLP dan Sentiment Analysis.

**VFLIX-APP** adalah aplikasi web interaktif yang membantu pengguna menjelajahi dunia perfilman melalui lensa data. Menggunakan teknik **Natural Language Processing (NLP)** dan **Sentiment Analysis**, aplikasi ini memberikan rekomendasi cerdas berdasarkan nuansa (*vibe*) atau alur cerita yang diinginkan pengguna.

---

## 🚀 Fitur Utama

* **🔎 Explore & Filter:** Menjelajahi katalog film dengan filter genre, pencarian judul, dan pengurutan berdasarkan sentimen penonton.
* **⚙️ Search with AI (Vibe-Based):** Fitur unggulan menggunakan model **TF-IDF** dan **Cosine Similarity**. Pengguna cukup menceritakan suasana film yang dicari, dan AI akan mencarikan kecocokan alur cerita.
* **🎥 Smart Trailer Integration:** Integrasi dengan **TMDB API** untuk memutar trailer YouTube langsung di dalam aplikasi.
* **📊 Data Insight:** Visualisasi distribusi genre film untuk memberikan gambaran tren industri film.

---

## 🛠️ Tech Stack

Aplikasi ini dibangun menggunakan ekosistem Python yang modern:
* **Frontend:** [Streamlit](https://streamlit.io/)
* **Data Processing:** [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
* **Machine Learning:** [Scikit-learn](https://scikit-learn.org/) (Model TF-IDF & Cosine Similarity)
* **External API:** [TMDB API](https://www.themoviedb.org/) untuk metadata film dan video trailer.

---

## 📂 Struktur Direktori

```text
.
├── data/               # Dataset film dan review
├── models/             # Pickle file untuk model AI (tfidf_vectorizer, matrix)
├── app.py              # File utama aplikasi Streamlit
└── requirements.txt    # Daftar dependensi library

```

---

## ⚙️ Cara Menjalankan Aplikasi

1. **Clone Repositori**
```bash
git clone [https://github.com/username/vflix-app.git](https://github.com/username/vflix-app.git)
cd vflix-app

```


2. **Instal Dependensi**
```bash
pip install -r requirements.txt

```


3. **Jalankan Aplikasi**
```bash
streamlit run app.py

```



---

## 👨‍💻 Tim Pengembang

Proyek ini dikembangkan oleh:

* **Hanif**
* **Reda**
* **Nabilah**

---