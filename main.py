import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pytesseract

# =========================================================
# PENGATURAN TESSERACT OCR
# =========================================================
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# =========================================================
# KONFIGURASI PROGRAM
# =========================================================
INPUT_FOLDER = "citraTTD"
OUTPUT_FOLDER = "hasil"

# Nilai Global Threshold
GLOBAL_THRESHOLD = 127

# Batas minimal rasio foreground untuk mendeteksi tanda tangan
# Diturunkan dari 2.0% menjadi 0.5% agar citra low contrast
# tetap dapat terdeteksi.
MIN_SIGNATURE_RATIO = 0.5

# Kernel untuk operasi morfologi
MORPH_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (3, 3)
)

# =========================================================
# DAFTAR FOLDER OUTPUT
# =========================================================
OUTPUT_FOLDERS = [
    "1_citra_asli",
    "2_grayscale",
    "3_global_thresh",
    "4_global_opening",
    "5_global_closing",
    "6_otsu_thresh",
    "7_otsu_opening",
    "8_otsu_closing",
    "9_perbandingan"
]

# Membuat folder output secara otomatis
for nama_folder in OUTPUT_FOLDERS:
    folder_path = os.path.join(
        OUTPUT_FOLDER,
        nama_folder
    )

    os.makedirs(
        folder_path,
        exist_ok=True
    )


# =========================================================
# FUNGSI OPERASI MORFOLOGI
# =========================================================
def proses_morfologi(binary_image):

    # Opening
    opening = cv2.morphologyEx(
        binary_image,
        cv2.MORPH_OPEN,
        MORPH_KERNEL
    )

    # Closing
    closing = cv2.morphologyEx(
        opening,
        cv2.MORPH_CLOSE,
        MORPH_KERNEL
    )

    return opening, closing


# =========================================================
# FUNGSI MEMPROSES SATU GAMBAR
# =========================================================
def proses_gambar(file_name):

    # -----------------------------------------------------
    # Membaca gambar
    # -----------------------------------------------------
    image_path = os.path.join(
        INPUT_FOLDER,
        file_name
    )

    image = cv2.imread(image_path)

    if image is None:
        print(
            f"Gagal membaca gambar: {file_name}"
        )
        return None

    # -----------------------------------------------------
    # 1. KONVERSI GRAYSCALE
    # -----------------------------------------------------
    gray_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # -----------------------------------------------------
    # 2. GLOBAL THRESHOLDING
    # -----------------------------------------------------
    _, global_binary = cv2.threshold(
        gray_image,
        GLOBAL_THRESHOLD,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Global Opening dan Closing
    global_opening, global_closing = proses_morfologi(
        global_binary
    )

    # -----------------------------------------------------
    # 3. OTSU THRESHOLDING
    # -----------------------------------------------------
    otsu_value, otsu_binary = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Otsu Opening dan Closing
    otsu_opening, otsu_closing = proses_morfologi(
        otsu_binary
    )

    # -----------------------------------------------------
    # 4. PERHITUNGAN FOREGROUND
    # -----------------------------------------------------
    jumlah_pixel = (
        gray_image.shape[0] *
        gray_image.shape[1]
    )

    # Foreground Global
    global_foreground = cv2.countNonZero(
        global_closing
    )

    # Foreground Otsu
    otsu_foreground = cv2.countNonZero(
        otsu_closing
    )

    # Rasio foreground Global
    global_ratio = (
        global_foreground /
        jumlah_pixel
    ) * 100

    # Rasio foreground Otsu
    otsu_ratio = (
        otsu_foreground /
        jumlah_pixel
    ) * 100

    # -----------------------------------------------------
    # 5. MENENTUKAN STATUS TANDA TANGAN
    # -----------------------------------------------------
    #
    # Perbaikan:
    # - Batas Global diturunkan dari 0.5 menjadi 0.1
    # - Batas Otsu diturunkan menjadi 0.5
    #
    # Tujuannya agar tanda tangan dengan kontras rendah
    # seperti gambar LowContrast tidak langsung dianggap
    # sebagai gambar kosong.
    # -----------------------------------------------------

    if (
        global_ratio < 0.1
        or (
            otsu_value > 220
            and otsu_ratio > 15.0
        )
    ):

        status = "SIGNATURE ABSENT"

        final_foreground = global_foreground
        final_ratio = global_ratio

    else:

        if otsu_ratio > MIN_SIGNATURE_RATIO:
            status = "SIGNATURE PRESENT"
        else:
            status = "SIGNATURE ABSENT"

        final_foreground = otsu_foreground
        final_ratio = otsu_ratio

    # -----------------------------------------------------
    # 6. MENYIMPAN HASIL PROSES
    # -----------------------------------------------------

    # Citra asli
    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            "1_citra_asli",
            file_name
        ),
        image
    )

    # Grayscale
    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            "2_grayscale",
            file_name
        ),
        gray_image
    )

    # Global Threshold
    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            "3_global_thresh",
            file_name
        ),
        global_binary
    )

    # Global Opening
    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            "4_global_opening",
            file_name
        ),
        global_opening
    )

    # Global Closing
    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            "5_global_closing",
            file_name
        ),
        global_closing
    )

    # Otsu Threshold
    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            "6_otsu_thresh",
            file_name
        ),
        otsu_binary
    )

    # Otsu Opening
    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            "7_otsu_opening",
            file_name
        ),
        otsu_opening
    )

    # Otsu Closing
    cv2.imwrite(
        os.path.join(
            OUTPUT_FOLDER,
            "8_otsu_closing",
            file_name
        ),
        otsu_closing
    )

    # -----------------------------------------------------
    # 7. MEMBUAT GAMBAR PERBANDINGAN
    # -----------------------------------------------------

    tinggi, lebar = gray_image.shape

    rasio_gambar = lebar / tinggi

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(20, 20 / rasio_gambar)
    )

    fig.suptitle(
        f"Analisis Deteksi Tanda Tangan: {file_name}",
        fontsize=13,
        fontweight="bold"
    )

    # =====================================================
    # BARIS PERTAMA
    # =====================================================

    # Citra asli
    axes[0, 0].imshow(
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
    )

    axes[0, 0].set_title(
        "Citra Asli",
        fontsize=10
    )

    # Grayscale
    axes[0, 1].imshow(
        gray_image,
        cmap="gray"
    )

    axes[0, 1].set_title(
        "Grayscale",
        fontsize=10
    )

    # Global Threshold
    axes[0, 2].imshow(
        global_binary,
        cmap="gray"
    )

    axes[0, 2].set_title(
        f"Global Thresh (T={GLOBAL_THRESHOLD})",
        fontsize=10
    )

    # Global Opening
    axes[0, 3].imshow(
        global_opening,
        cmap="gray"
    )

    axes[0, 3].set_title(
        "Global Opening",
        fontsize=10
    )

    # Global Closing
    axes[0, 4].imshow(
        global_closing,
        cmap="gray"
    )

    axes[0, 4].set_title(
        "Global Closing",
        fontsize=10
    )

    # =====================================================
    # BARIS KEDUA
    # =====================================================

    # Otsu Threshold
    axes[1, 0].imshow(
        otsu_binary,
        cmap="gray"
    )

    axes[1, 0].set_title(
        f"Otsu Thresh (T={int(otsu_value)})",
        fontsize=10
    )

    # Otsu Opening
    axes[1, 1].imshow(
        otsu_opening,
        cmap="gray"
    )

    axes[1, 1].set_title(
        "Otsu Opening",
        fontsize=10
    )

    # Otsu Closing
    axes[1, 2].imshow(
        otsu_closing,
        cmap="gray"
    )

    axes[1, 2].set_title(
        "Otsu Closing",
        fontsize=10
    )

    # Panel kosong
    axes[1, 3].axis("off")

    # =====================================================
    # PANEL STATUS
    # =====================================================

    axes[1, 4].axis("off")

    if status == "SIGNATURE PRESENT":
        status_color = "green"
    else:
        status_color = "red"

    # Status
    axes[1, 4].text(
        0.5,
        0.65,
        status,
        color=status_color,
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.5",
            ec=status_color,
            fc="none",
            lw=2
        )
    )

    # Jumlah foreground
    axes[1, 4].text(
        0.5,
        0.40,
        f"Piksel: {final_foreground} px",
        fontsize=10,
        ha="center",
        va="center"
    )

    # Rasio foreground
    axes[1, 4].text(
        0.5,
        0.25,
        f"Rasio: {final_ratio:.2f}%",
        fontsize=10,
        ha="center",
        va="center"
    )

    # -----------------------------------------------------
    # MENGHILANGKAN SUMBU KOORDINAT
    # -----------------------------------------------------
    for baris in axes:

        for ax in baris:

            if ax not in [
                axes[1, 3],
                axes[1, 4]
            ]:
                ax.axis("off")

    plt.tight_layout()

    # -----------------------------------------------------
    # 8. MENYIMPAN GAMBAR PERBANDINGAN
    # -----------------------------------------------------

    nama_dasar = os.path.splitext(
        file_name
    )[0]

    nama_hasil = (
        f"Perbandingan_{nama_dasar}.png"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_FOLDER,
            "9_perbandingan",
            nama_hasil
        ),
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # -----------------------------------------------------
    # DATA UNTUK TABEL REKAP
    # -----------------------------------------------------

    return {
        "filename": file_name,
        "global_t": GLOBAL_THRESHOLD,
        "otsu_t": int(otsu_value),
        "fg_px": final_foreground,
        "pct": final_ratio,
        "status": status
    }


# =========================================================
# PROGRAM UTAMA
# =========================================================
def main():

    # -----------------------------------------------------
    # CEK FOLDER INPUT
    # -----------------------------------------------------

    if not os.path.isdir(INPUT_FOLDER):

        print(
            f"Folder '{INPUT_FOLDER}' tidak ditemukan!"
        )

        return

    # -----------------------------------------------------
    # MENCARI FILE GAMBAR
    # -----------------------------------------------------

    image_files = sorted(
        [
            file
            for file in os.listdir(INPUT_FOLDER)
            if file.lower().endswith(
                (
                    ".png",
                    ".jpg",
                    ".jpeg"
                )
            )
        ]
    )

    # -----------------------------------------------------
    # CEK APAKAH ADA GAMBAR
    # -----------------------------------------------------

    if len(image_files) == 0:

        print(
            f"Tidak terdapat gambar di folder "
            f"'{INPUT_FOLDER}'."
        )

        return

    # List untuk menyimpan hasil
    hasil_rekap = []

    # =====================================================
    # HEADER PROGRAM
    # =====================================================

    print("=" * 105)

    print(
        "PROSES ANALISIS MORFOLOGI & DETEKSI TANDA TANGAN "
        "(DUAL PATH: GLOBAL & OTSU)"
    )

    print("=" * 105)

    # =====================================================
    # PROSES SELURUH GAMBAR
    # =====================================================

    for nama_file in image_files:

        hasil = proses_gambar(
            nama_file
        )

        if hasil is not None:

            hasil_rekap.append(
                hasil
            )

            print(
                f"Diproses: {nama_file:<40} "
                f"-> Status: {hasil['status']}"
            )

    # =====================================================
    # REKAPITULASI HASIL
    # =====================================================

    print("\n" + "=" * 105)

    print(
        "                         REKAPITULASI HASIL SEGMENTASI & DETEKSI"
    )

    print("=" * 105)

    # Header tabel
    print(
        f"{'No':<4} | "
        f"{'Nama Gambar':<35} | "
        f"{'Global T':<9} | "
        f"{'Otsu T':<8} | "
        f"{'Piksel FG':<10} | "
        f"{'Rasio (%)':<9} | "
        f"{'Status':<18}"
    )

    print("-" * 105)

    # Isi tabel
    for nomor, data in enumerate(
        hasil_rekap,
        1
    ):

        print(
            f"{nomor:<4} | "
            f"{data['filename'][:34]:<35} | "
            f"{data['global_t']:<9} | "
            f"{data['otsu_t']:<8} | "
            f"{data['fg_px']:<10} | "
            f"{data['pct']:<9.2f} | "
            f"{data['status']:<18}"
        )

    print("=" * 105)

    # =====================================================
    # PESAN SELESAI
    # =====================================================

    print(
        f"\n[SELESAI] Hasil analisis telah disimpan di "
        f"'{OUTPUT_FOLDER}/9_perbandingan/'."
    )


# =========================================================
# MENJALANKAN PROGRAM
# =========================================================
if __name__ == "__main__":
    main()