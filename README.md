# 🍿 VFLIX-APP  
### AI-Based Movie Recommendation Platform

<p align="center">
  <img src="assets/image.png" alt="VFLIX Banner" width="850"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?logo=scikitlearn">
  <img src="https://img.shields.io/badge/API-TMDB-green">
  <img src="https://img.shields.io/badge/Status-Active-success">
</p>

---

## 📖 About The Project

**VFLIX-APP** adalah platform rekomendasi film berbasis Artificial Intelligence yang dirancang untuk membantu pengguna menemukan film sesuai dengan *vibe*, suasana, atau alur cerita yang diinginkan.

Aplikasi ini memanfaatkan teknologi **Natural Language Processing (NLP)** dan **Sentiment Analysis** untuk menghasilkan rekomendasi yang lebih personal dan relevan. Pengguna dapat mencari film menggunakan deskripsi bebas seperti:

> *"good superman film"*  
> *"amazing scifi film"*

Sistem kemudian akan mencocokkan deskripsi tersebut menggunakan metode **TF-IDF** dan **Cosine Similarity**.

---

# ✨ Features

## 🔎 Explore & Filter
- Menjelajahi katalog film interaktif
- Filter berdasarkan genre
- Search judul film
- Sorting berdasarkan sentiment score

## ⚙️ AI Recommendation System
- Rekomendasi berbasis deskripsi teks (*vibe-based search*)
- Menggunakan:
  - TF-IDF Vectorization
  - Cosine Similarity
  - NLP preprocessing

## 🎥 Smart Trailer Integration
- Integrasi dengan **TMDB API**
- Menampilkan trailer YouTube langsung di aplikasi

## 📊 Data Visualization
- Visualisasi distribusi genre film
- Insight sederhana mengenai tren perfilman

---

# 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python |
| Data Processing | Pandas |
| Machine Learning | Scikit-learn |
| NLP | TF-IDF |
| Model Persistence | Joblib |
| API Integration | TMDB API |

---

# 📂 Project Structure

```plaintext
├── .devcontainer/              # Konfigurasi container development
│   └── devcontainer.json       # Pengaturan environment VS Code Dev Container

├── .streamlit/                 # Konfigurasi Streamlit
│   └── config.toml             # Pengaturan tampilan & server Streamlit

├── assets/                     # Folder asset/gambar pendukung
│   └── image.png               # Preview atau banner aplikasi

├── data/                       # Dataset utama aplikasi
│   ├── cast.csv                # Data pemeran film
│   ├── crew.csv                # Data kru film
│   ├── genres.csv              # Data genre film
│   ├── movies.csv              # Dataset film mentah
│   ├── movies_scored_final.csv # Dataset film hasil scoring/sentiment
│   ├── reviews.csv             # Dataset review mentah
│   └── reviews_final.csv       # Dataset review hasil preprocessing

├── models/                     # Model Machine Learning tersimpan
│   ├── sentiment_model.pkl     # Model analisis sentimen
│   ├── tfidf_matrix.pkl        # Matrix TF-IDF hasil training
│   ├── tfidf_rekomendasi.pkl   # Model rekomendasi berbasis TF-IDF
│   └── tfidf_vectorizer.pkl    # TF-IDF Vectorizer

├── app.py                      # File utama aplikasi Streamlit
├── README.md                   # Dokumentasi proyek
└── requirements.txt            # Daftar dependensi Python
```

---

## 📓 Jupyter Notebook

[Click here to open the notebook](https://colab.research.google.com/drive/1GOuEtwZLgqbmRGwfs17vIdhakZ56ndb-?usp=sharing)

---

# 🚀 Live Demo

🔗 https://vflix-app.streamlit.app

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/marcoreda2007-lgtm/vflix-app.git
cd vflix-app
```

---

## 2️⃣ Install Dependencies

Pastikan Python sudah terinstall.

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Run The Application

```bash
streamlit run app.py
```

---

# 🔑 Environment Variables

Aplikasi membutuhkan API Key dari TMDB.

Buat file `.env` lalu tambahkan:

```env
TMDB_API_KEY=your_api_key_here
```

---

# 📸 Application Preview

<p align="center">
  <img src="assets/preview1.png" width="45%">
  <img src="assets/preview2.png" width="45%">
</p>

---

# 👨‍💻 Developer Team

| Name |
|---|
| Hanif |
| Reda |
| Nabilah |

---

# 📌 Notes

- Pastikan `TMDB_API_KEY` aktif agar fitur trailer berjalan normal.
- Direkomendasikan menggunakan Python 3.10+.
- Dataset yang digunakan telah melalui proses preprocessing dan sentiment scoring.

---

# ⭐ Support

Jika project ini membantu, jangan lupa kasih ⭐ di repository GitHub ya!

---
