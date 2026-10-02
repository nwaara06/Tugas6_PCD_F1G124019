# Mini Project PCD - Deteksi Tanda Tangan

## Deskripsi
Mini project ini merupakan implementasi pengolahan citra digital untuk mendeteksi keberadaan tanda tangan pada dokumen.

Tahapan pengolahan citra yang digunakan meliputi:
* Crop area tanda tangan
* Konversi citra ke grayscale
* Global Thresholding
* Otsu Thresholding
* Operasi morfologi Opening
* Operasi morfologi Closing
* Perhitungan jumlah foreground pixel
* Penentuan keberadaan tanda tangan

Pada project ini, area tanda tangan yang dianalisis adalah tanda tangan Dekan pada dokumen ijazah universitas.

---

## Teknologi yang Digunakan
* Python
* OpenCV
* NumPy
* Matplotlib

---

## How to Run

### 1. Clone Repository
Buka Command Prompt atau PowerShell, kemudian jalankan:
```bash
git clone https://github.com/nwaara06/Tugas6_PCD_F1G124019.git
```

Kemudian masuk ke folder project:
```bash
cd Tugas6_PCD_F1G124019
```

### 2. Install Library
Install library yang diperlukan dengan perintah:
```bash
pip install opencv-python numpy matplotlib
```

### 3. Jalankan Program
Jalankan program dengan:
```bash
python main.py
```

---

## Hasil Pengolahan

Program melakukan beberapa tahap pengolahan:
* Grayscale
* Global Thresholding
* Otsu Thresholding
* Opening
* Closing

Hasil pengolahan disimpan pada folder:
`hasil/`

---

## Metode yang Digunakan

* **Global Thresholding**  
  Global threshold digunakan untuk memisahkan foreground dan background menggunakan nilai ambang tertentu. Pada program ini digunakan nilai threshold 127.

* **Otsu Thresholding**  
  Metode Otsu menentukan nilai threshold secara otomatis berdasarkan distribusi intensitas citra.

* **Opening**  
  Opening digunakan untuk mengurangi noise kecil pada hasil thresholding.

* **Closing**  
  Closing digunakan untuk membantu menghubungkan bagian foreground yang terputus dan mengisi celah kecil.

---

## Output

Output program berupa:
* Citra hasil crop
* Citra grayscale
* Hasil Global Thresholding
* Hasil Otsu Thresholding
* Hasil Opening
* Hasil Closing
* Jumlah foreground pixel
* Status keputusan (SIGNATURE PRESENT / SIGNATURE ABSENT)
* File `rekapitulasi_hasil.csv`

---

## Dataset

Dataset yang digunakan terdiri dari beberapa citra dokumen ijazah dengan kondisi kualitas citra yang berbeda.  
Contohnya:
* High Quality
* Low Contrast
* Blurred
* High Noise
* Low Resolution
* Faded / Underexposed
* Color Shift
* JPEG Compression
* Combined Degradation

---

## Author

* **Nama:** Wa Rahmawati
* **NIM:** F1G124019
* **Mata Kuliah:** Pengolahan Citra Digital