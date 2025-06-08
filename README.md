**Amazon Kindle Store Reviews - Binary Sentiment Analysis**

Tugas Besar Mata Kuliah Big Data & AI
1304221056 - Naufal Gholib Shiddiq  - IF-45-GABPJJ

---

## 📖 Deskripsi

Proyek ini melakukan analisis sentimen biner pada dataset Amazon Kindle Store Review menggunakan Apache PySpark. Review dengan rating 1-2 diklasifikasikan sebagai Sentimen Negatif, dan rating 4-5 sebagai Sentimen Positif. Proses meliputi data loading, preprocessing, EDA, feature engineering, pelatihan model, evaluasi, hingga real-time prediction untuk testing model ketika di inputkan user berbagai contoh kalimat review.

## 🗂️ Tentang Dataset

### **Konteks**

Dataset review produk dari kategori [Amazon Kindle Store](https://www.kaggle.com/datasets/bharadwaj6/kindle-reviews)

### **Isi Dataset**

Dataset [5-core](https://en.wikipedia.org/wiki/Degeneracy_(graph_theory)) review produk dari kategori Amazon Kindle Store dari Mei 1996 - Juli 2014. Berisi total 982619 baris data. Setiap reviewer memiliki setidaknya 5 review dan setiap produk memiliki setidaknya 5 review dalam dataset ini.

#### **Kolom**

* `asin` - ID dari product yang dijual, like B000FA64PK
* `helpful` - rating manfaat dari ulasan tersebut - contoh: 2/3.
* `overall` - rating dari product 
* `reviewText` - text dari review.
* `reviewTime` - date and time review dibuat.
* `reviewerID` - ID dari reviewer, contohnya A3SPTOKDG7WBLN
* `reviewerName` - nama reviewer.
* `summary` - keismpulan dari review (deskripsi).
* `unixReviewTime` - unix timestamp.

## 📂 Struktur Proyek

```
├── README.md                                            # Dokumentasi proyek
├── 1304221056_Final_Project_Big_Data_&_AI.ipynb         # Notebook eksplorasi (jika ada)
├── sentiment_analysis.py                                # Source Code
│   
```

## 🛠️ Prasyarat

* Python 3.8+
* Java 8+ (untuk Spark)
* Apache Spark 3.x
* Pandas, NumPy, Matplotlib, Seaborn, Plotly
* TextBlob

## 📥 Instalasi

1. **Clone repository**

   ```bash
   git clone https://github.com/naufalgholib/sentiment-analysis.git
   cd sentiment-analysis
   ```
2. **Atur virtual environment**

   ```bash
   python -m venv sentinment-analysis
   source sentinment-analysis/bin/activate  # Linux/macOS
   sentinment-analysis\\Scripts\\activate  # Windows
   ```
3. **Install dependencies**

   ```bash
   pip install pandas numpy matplotlib seaborn plotly textblob pyspark
   ```
4. **Download dataset**

   * Manual download dari Kaggle: [https://www.kaggle.com/datasets/bharadwaj6/kindle-reviews](https://www.kaggle.com/datasets/bharadwaj6/kindle-reviews)
   * [Simpan ](https://www.kaggle.com/datasets/bharadwaj6/kindle-reviews)[`kindle_reviews.csv`](https://www.kaggle.com/datasets/bharadwaj6/kindle-reviews)[ di folder ](https://www.kaggle.com/datasets/bharadwaj6/kindle-reviews)[`yang sama dengan source code`](https://www.kaggle.com/datasets/bharadwaj6/kindle-reviews)[ atau gun](https://www.kaggle.com/datasets/bharadwaj6/kindle-reviews)akan perintah curl:

     ```bash
     curl -O -L "https://www.kaggle.com/api/v1/datasets/download/bharadwaj6/kindle-reviews"
     unzip kindle-reviews
     ```

## 🚀 Cara Menjalankan

Jalankan skrip Python utama:

```bash
python sentiment_analysis.py
```

Atau, jika di notebook:

```bash
jupyter notebook
```

## 📝 Workflow

1. **Import Libraries**: Memuat semua library (PySpark, Pandas, Plotly, TextBlob, dll.)
2. **Spark Session**: Inisialisasi `SparkSession` dengan konfigurasi optimal.
3. **Loading Data**: Download dan ekstrak dataset, kemudian load ke DataFrame Spark.
4. **Data Cleaning**:

   * Hilangkan null dan pengamatan kosong.
   * Filter rating 1-2 (negatif) & 4-5 (positif).
   * Gabungkan `summary` dan `reviewText`, buat label sentimen biner.
   * Konversi timestamp review.
5. **Exploratory Data Analysis (EDA)**:

   * Distribusi rating dan sentimen.
   * Tren ulasan per waktu.
   * Visualisasi interaktif dengan Plotly.
6. **Text Preprocessing**:

   * Pembersihan teks (hapus HTML, special chars, lowercasing) via UDF PySpark.
7. **Feature Engineering**:

   * Tokenization, stopwords removal.
   * Bag-of-words (CountVectorizer) + TF-IDF.
8. **Training Model**:

   * Split data (80/20).
   * Latih Logistic Regression & Random Forest.
9. **Evaluasi Model**:

   * Hitung accuracy, F1-score, precision, recall, AUC-ROC.
   * Bandingkan performa model.
   * Visualisasi perbandingan metrik.
10. **Detail Analysis**:

    * Pilih model terbaik berdasarkan AUC-ROC.
    * Confusion matrix & classification report.
11. **Business Insights**:

    * Analisis sentimen per produk (ASIN).
    * Korrelasi rating vs prediksi sentimen.
    * Contoh kesalahan klasifikasi.
12. **Advanced Visualizations**:

    * Distribusi aktual vs prediksi.
    * Rating vs prediksi sentimen.
13. **Real-Time Prediction**:

    * Fungsi `predict_binary_sentiment(text)` untuk inferensi on-the-fly.
14. **Summary & Kesimpulan**:

    * Laporan ringkas dataset, performa model, dan key insight.

## 📊 Hasil Utama

* Model terbaik: **Logistic Regression**
* Accuracy: 0.970
* F1-Score: 0.968
* Precision: 0.968
* Recall: 0.970
* Insight: 94.5% of reviews are predicted as Positive dan 5.5% of reviews are predicted as Negative
---
