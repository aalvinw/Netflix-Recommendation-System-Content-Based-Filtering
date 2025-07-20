

# Laporan Proyek Machine Learning - Agustinus Alvin

**Netflix Recommendation System: Content-Based Filtering**

---

## Project Overview

Dalam era layanan streaming, Netflix menjadi salah satu platform terpopuler yang menyediakan ribuan film dan serial TV. Namun, jumlah konten yang sangat banyak seringkali membuat pengguna kesulitan dalam memilih tontonan yang sesuai dengan preferensi mereka.

Sistem rekomendasi hadir sebagai solusi untuk membantu pengguna menemukan film yang relevan dengan selera mereka. Proyek ini bertujuan untuk membangun sistem rekomendasi berbasis konten (content-based filtering) yang dapat merekomendasikan film mirip berdasarkan deskripsi dan genre.

### Rubrik Tambahan (Opsional)

Menurut studi oleh Ricci et al. (2015) dalam buku *Recommender Systems Handbook*, sistem rekomendasi sangat berperan dalam meningkatkan user engagement dan konversi dalam platform digital seperti Netflix.

---

## Business Understanding

### Problem Statements

1. Bagaimana cara merekomendasikan film yang mirip dengan film tertentu berdasarkan genre dan deskripsi?
2. Bagaimana meningkatkan kepuasan pengguna Netflix dengan sistem rekomendasi yang personal?

### Goals

1. Mengembangkan sistem rekomendasi berbasis content-based filtering untuk memberikan daftar film yang mirip dengan film input.
2. Meningkatkan pengalaman pengguna dalam menemukan tontonan yang relevan.

### Solution Approach

**Solution Statement:**

* Menggunakan pendekatan **Content-Based Filtering** untuk menyarankan film berdasarkan kesamaan fitur konten.
* Menghitung kemiripan antar film menggunakan **TF-IDF** pada deskripsi dan genre, serta **Cosine Similarity**.

**Algoritma yang digunakan:**

* TF-IDF Vectorizer (untuk kolom `description` dan `listed_in`)
* Cosine Similarity (untuk mengukur kemiripan antar film)

---

## Data Understanding

Dataset digunakan dari Kaggle dengan judul [Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows). Dataset yang digunakan berjudul Netflix Movies and TV Shows dan terdiri dari 8.807 baris dan 12 kolom. Dataset ini memuat informasi mengenai film dan acara TV yang tersedia di platform Netflix.

Fitur utama dalam dataset:

| Nama Kolom     | Deskripsi                                                         |
| -------------- | ----------------------------------------------------------------- |
| `show_id`      | ID unik tiap konten                                               |
| `type`         | Jenis konten: *Movie* atau *TV Show*                              |
| `title`        | Judul film atau acara TV                                          |
| `director`     | Nama sutradara                                                    |
| `cast`         | Daftar aktor/aktris                                               |
| `country`      | Negara asal produksi                                              |
| `date_added`   | Tanggal ketika konten ditambahkan ke Netflix                      |
| `release_year` | Tahun rilis dari konten                                           |
| `rating`       | Kategori usia (contoh: PG, TV-MA, R, dll)                         |
| `duration`     | Durasi konten (menit untuk film, atau jumlah musim untuk TV Show) |
| `listed_in`    | Genre/kategori konten (misal: Action, Drama, Comedy)              |
| `description`  | Deskripsi singkat dari konten                                     |



### Kondisi Missing Values

Dari total 8.807 baris data, beberapa kolom memiliki nilai kosong (missing value) sebagai berikut:

| Kolom        | Jumlah Nilai Kosong |
| ------------ | ------------------- |
| `director`   | 2.634               |
| `cast`       | 825                 |
| `country`    | 831                 |
| `date_added` | 10                  |
| `rating`     | 4                   |
| `duration`   | 3                   |
| Kolom lain   | 0                   |

### **Nilai Duplikat**:

   * Tidak ada baris duplikat (`0`).

### **Kapasitas Memori**:

   * Dataset menggunakan sekitar **825 KB**, cukup ringan untuk pemrosesan lokal.

### Rubrik Tambahan

Fitur description dan listed_in yang menjadi fokus utama sistem rekomendasi tidak memiliki nilai kosong sehingga aman digunakan langsung. Sementara fitur lain dengan missing values ditangani pada tahap persiapan data (Data Preparation) dengan imputasi sederhana atau pengisian nilai default seperti 'unknown'.
---


## 3. Data Preparation

Terima kasih, penjelasanmu tentang *data preparation* di notebook sudah sangat sistematis dan lengkap. Sekarang mari kita **revisi bagian "Data Preparation" di laporan** agar benar-benar **selaras** dengan langkah-langkah di notebook dan **memenuhi kriteria penilaian**.

Berikut ini adalah versi revisi yang disesuaikan:

---

## 3. Data Preparation

Sebelum membangun sistem rekomendasi berbasis konten, dilakukan beberapa langkah persiapan data sebagai berikut:

### 3.1 Ekstraksi dan Transformasi Data

1. **Salin Dataset**
   Dataset asli disalin ke variabel baru (`df_prep`) menggunakan `.copy()` untuk menjaga data mentah tetap utuh selama proses manipulasi.

2. **Ekstraksi Genre dari Kolom `listed_in`**
   Kolom `listed_in`, yang berisi daftar genre dalam bentuk string, diubah menjadi list menggunakan pemisah koma (`,`). Hasilnya disimpan dalam kolom baru bernama `genres_list`.

3. **Penanganan Nilai Kosong**
   Kolom `description` diisi dengan string kosong (`''`) menggunakan `fillna()` untuk menghindari error saat proses ekstraksi fitur berbasis teks.

4. **Seleksi Kolom**
   Kolom-kolom yang digunakan untuk membangun sistem rekomendasi meliputi: `title`, `genres_list`, `description`, `cast`, `director`, dan `type`.

### 3.2 Rekayasa Fitur (Feature Engineering)

1. **Binarisasi Genre**
   Fitur `genres_list` diubah menjadi representasi numerik biner menggunakan **MultiLabelBinarizer**, menghasilkan matriks `genre_mat` di mana setiap genre menjadi fitur biner (1 atau 0).

2. **Ekstraksi Fitur Teks dari Deskripsi (description)**
   Deskripsi tayangan diubah menjadi vektor numerik menggunakan **TfidfVectorizer**:

   * Stopwords dihapus dengan `stop_words='english'`
   * Jumlah fitur dibatasi hingga 5000 kata terpenting (`max_features=5000`)
   * Hasilnya disimpan dalam `desc_mat`

3. **Ekstraksi Fitur dari Pemeran (cast) dan Sutradara (director)**
   Kedua kolom tersebut dianggap sebagai fitur teks karena nama-nama yang muncul bisa memberi sinyal gaya atau karakteristik tayangan.

   * Masing-masing dikonversi menjadi teks (jika list), lalu diubah ke bentuk vektor menggunakan **TfidfVectorizer**
   * Hasilnya disimpan sebagai `cast_mat` dan `director_mat`

4. **Penggabungan Semua Fitur**
   Semua fitur numerik (`genre_mat`, `desc_mat`, `cast_mat`, dan `director_mat`) digabungkan secara horizontal menggunakan `hstack()` dari SciPy, membentuk `feature_mat`, yaitu matriks representasi lengkap setiap tayangan.

### 3.3 Perhitungan Kemiripan

Langkah terakhir adalah menghitung **Cosine Similarity** antar baris dalam `feature_mat` untuk mengukur seberapa mirip satu tayangan dengan tayangan lainnya berdasarkan gabungan fitur genre, deskripsi, pemeran, dan sutradara.



---

## Modeling
Berikut adalah **revisi bagian Modeling & Results** pada laporan agar **konsisten dengan output notebook**, sesuai saran dari reviewer. Judul film yang digunakan dalam contoh diubah menjadi **"Ganglands"**, agar sesuai dengan hasil yang benar-benar dihasilkan oleh notebook:
### 1 Rekomendasi Berdasarkan Deskripsi dan Genre

**(TF-IDF + Cosine Similarity)**

#### 🔍 Pendekatan:

* Dua kolom utama digunakan sebagai sumber fitur konten:

  * `listed_in` (genre)
  * `description` (deskripsi naratif)
* Kolom-kolom tersebut digabung menjadi satu kolom teks `combined_features` untuk membentuk representasi konten yang lebih kaya.
* Fitur `combined_features` kemudian diubah menjadi vektor numerik menggunakan **TF-IDF Vectorizer**.
* Kemiripan antar tayangan dihitung menggunakan **Cosine Similarity** terhadap hasil vektor tersebut.

####  Hasil:

* Sistem berhasil memberikan rekomendasi tayangan yang memiliki kemiripan dari segi genre dan deskripsi naratif.
* Contoh:

  * Untuk input **"Ganglands"**, sistem merekomendasikan beberapa tayangan serupa berdasarkan kemiripan konten, seperti:
             title  \
  6741           Fatal Destiny   
  3976  The Eagle of El-Se'eed   
  734                    Lupin   
  543               Undercover   
  5194               The Truth   
  2676                   Fauda   
  11          Bangkok Breaking    
  3414                  Chosen   
  4662            Monkey Twins   
  4752                 Smoking  
  * Tayangan-tayangan tersebut memiliki kemiripan dalam genre kejahatan, drama, dan narkotika.

---

### 2 Alternatif Modeling (Opsional)

> Pendekatan ini belum diimplementasikan dalam proyek saat ini, namun dapat dijadikan pengembangan lanjutan.

#### 👥 Collaborative Filtering:

* Cocok digunakan apabila tersedia data historis pengguna (misalnya: riwayat tontonan, rating, atau preferensi).
* Pendekatan ini merekomendasikan tayangan berdasarkan pola kesamaan antar pengguna atau antar item.
* Memberikan potensi untuk sistem rekomendasi yang lebih personal.

Karena dataset tidak menyertakan data pengguna, maka proyek ini menggunakan pendekatan **Content-Based Filtering**.

---

## Evaluation


###  Analisis Evaluasi Rekomendasi untuk “Ganglands”

#### 1. Distribusi Skor Cosine Similarity

Berdasarkan visualisasi **Distribusi Skor Cosine Similarity** untuk tayangan **“Ganglands”**, diperoleh beberapa insight berikut:

1. **Skor Similarity yang Moderat**

   * Nilai cosine similarity berkisar antara **0.55 hingga 0.62**, menunjukkan adanya **kemiripan konten yang cukup kuat namun tidak ekstrem** antara “Ganglands” dan tayangan yang direkomendasikan.
   * Ini berarti model mampu menemukan tayangan dengan konten (genre dan deskripsi) yang berkaitan secara semantik.

2. **Tayangan Paling Mirip**

   * Tayangan **“Fatal Destiny”** memiliki similarity tertinggi (**0.62**), diikuti oleh “The Eagle of El-Se’eed” (**0.58**) dan “Lupin” (**0.57**).
   * Tayangan-tayangan tersebut memiliki tema **drama kriminal**, **ketegangan**, dan **kejahatan terorganisir**, yang secara naratif cocok dengan karakteristik "Ganglands".

3. **Mayoritas Tayangan Mendekati Rata-Rata**

   * Sebagian besar skor berada di sekitar **0.56**, menandakan bahwa tayangan yang direkomendasikan berada dalam **zona kemiripan yang konsisten** namun tidak terlalu dominan satu sama lain.

4. **Rentang Skor yang Relatif Sempit**

   * Perbedaan skor cosine similarity dari rekomendasi terendah ke tertinggi hanya sekitar **0.07 poin**, menunjukkan bahwa model cenderung memberikan rekomendasi yang **relatif homogen** dari segi konten.
   * Hal ini bisa dilihat sebagai stabilitas model, namun juga menjadi sinyal perlunya **penambahan keberagaman (diversity)** untuk meningkatkan eksplorasi pengguna terhadap konten baru.

---

#### 2. Precision\@5 untuk “Ganglands”

Sebagai pelengkap evaluasi kualitatif di atas, dilakukan pula evaluasi kuantitatif dengan metrik **Precision\@5**, yaitu proporsi tayangan yang benar-benar relevan dari 5 rekomendasi teratas.

* Kriteria relevansi: tayangan dianggap relevan jika memiliki **genre yang sama** dengan tayangan input.
* Hasil:

> **Precision\@5 = 1.00**

Ini berarti **semua dari 5 rekomendasi teratas** memiliki genre yang **relevan** dengan "Ganglands". Dengan kata lain, sistem rekomendasi berhasil menyarankan tayangan yang sesuai secara genre dalam 5 pilihan pertamanya.

---

### Kesimpulan Evaluasi

* Sistem rekomendasi berbasis konten berhasil memberikan rekomendasi yang **konsisten, relevan, dan stabil**, dengan skor similarity yang moderat dan Precision\@5 yang tinggi.
* Untuk pengembangan lanjutan, sistem bisa diperkaya dengan pendekatan hybrid atau strategi **diversifikasi** agar rekomendasi tidak hanya terbatas pada konten yang terlalu mirip, melainkan juga membuka kemungkinan eksplorasi terhadap konten baru namun tetap relevan.


---

## Referensi

1. Ricci, Rokach, Shapira. *Recommender Systems Handbook*, Springer, 2015.
2. Shivam Bansal. [Netflix Movies and TV Shows Dataset on Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)
3. Scikit-learn Documentation: [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
4. Towards Data Science: *Building Content-Based Recommendation System with Python*

