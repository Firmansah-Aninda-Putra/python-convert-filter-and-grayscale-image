import cv2
import numpy as np
import matplotlib.pyplot as plt

# Baca gambar
img = cv2.imread(r'C:\Users\bumii\Desktop\GRAYSCALE IMAGE\kapsul.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("TUGAS MORFOLOGI CITRA - DETEKSI KAPSUL MERAH")
print("=" * 50)

# ========== SOAL 1-4: METODE RGB ==========
print("\n--- METODE RGB (Soal 1-4) ---")

# 1. Split color Red
red = img_rgb[:, :, 0]
print("1. Red channel extracted ✓")

# 2. Convert ke grayscale
gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
print("2. Grayscale conversion ✓")

# 3. Red minus Grayscale
r_minus_gs = cv2.subtract(red, gray)
print("3. R - GS subtraction ✓")

# 4. Normalisasi
normalized = cv2.normalize(r_minus_gs, None, 0, 255, cv2.NORM_MINMAX)
print("4. Normalization ✓")

# ========== SOAL 5: METODE HSV (ALTERNATIF) ==========
print("\n--- METODE HSV (Soal 5) ---")

# 5. Gunakan HSV untuk ambil warna merah
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# Range merah di HSV (2 range karena hue merah di 0 dan 180)
mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
hsv_red = cv2.bitwise_or(mask1, mask2)
print("5. HSV red extraction ✓")

# ========== SOAL 6: THRESHOLD ==========
print("\n--- THRESHOLD (Soal 6) ---")

# 6. Threshold - pilih metode HSV karena lebih baik
binary = hsv_red
print("6. Binary thresholding ✓")

# ========== SOAL 7-9: OPERASI MORFOLOGI ==========
print("\n--- OPERASI MORFOLOGI (Soal 7-9) ---")

# Structuring element
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# 7. Erosi dan Dilasi terpisah
erosi = cv2.erode(binary, kernel)
dilasi = cv2.dilate(binary, kernel)
print("7. Erosion and Dilation ✓")

# 8. Erosi kemudian Dilasi (OPENING)
opening = cv2.erode(binary, kernel)
opening = cv2.dilate(opening, kernel)
print("8. Opening (Erosion → Dilation) ✓")

# 9. Dilasi kemudian Erosi (CLOSING)
closing = cv2.dilate(binary, kernel)
closing = cv2.erode(closing, kernel)
print("9. Closing (Dilation → Erosion) ✓")

# ========== VISUALISASI ==========
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('HASIL MORFOLOGI CITRA - DETEKSI KAPSUL', fontsize=14, fontweight='bold')

# Baris 1: Soal 1-4
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title('Original')
axes[0, 0].axis('off')

axes[0, 1].imshow(red, cmap='Reds')
axes[0, 1].set_title('1. Red Channel')
axes[0, 1].axis('off')

axes[0, 2].imshow(gray, cmap='gray')
axes[0, 2].set_title('2. Grayscale')
axes[0, 2].axis('off')

axes[0, 3].imshow(normalized, cmap='gray')
axes[0, 3].set_title('3-4. R-GS Normalized')
axes[0, 3].axis('off')

# Baris 2: Soal 5-6 dan hasil morfologi dasar
axes[1, 0].imshow(hsv_red, cmap='gray')
axes[1, 0].set_title('5-6. Binary (HSV)')
axes[1, 0].axis('off')

axes[1, 1].imshow(erosi, cmap='gray')
axes[1, 1].set_title('7. Erosion')
axes[1, 1].axis('off')

axes[1, 2].imshow(dilasi, cmap='gray')
axes[1, 2].set_title('7. Dilation')
axes[1, 2].axis('off')

axes[1, 3].imshow(binary, cmap='gray')
axes[1, 3].set_title('Binary (Base)')
axes[1, 3].axis('off')

# Baris 3: Soal 8-9
axes[2, 0].imshow(opening, cmap='gray')
axes[2, 0].set_title('8. Opening')
axes[2, 0].axis('off')

axes[2, 1].imshow(closing, cmap='gray')
axes[2, 1].set_title('9. Closing')
axes[2, 1].axis('off')

# Perbandingan hasil akhir
result_opening = cv2.bitwise_and(img_rgb, img_rgb, mask=opening)
axes[2, 2].imshow(result_opening)
axes[2, 2].set_title('Result: Opening')
axes[2, 2].axis('off')

result_closing = cv2.bitwise_and(img_rgb, img_rgb, mask=closing)
axes[2, 3].imshow(result_closing)
axes[2, 3].set_title('Result: Closing')
axes[2, 3].axis('off')

plt.tight_layout()
plt.savefig('hasil_morfologi.png', dpi=150, bbox_inches='tight')
print("\n✓ Hasil disimpan: hasil_morfologi.png")
plt.show()

# ========== SOAL 10: ANALISIS ==========
print("\n" + "=" * 50)
print("10. ANALISIS PERBANDINGAN")
print("=" * 50)

def hitung_objek(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def hitung_pixel(img):
    return np.sum(img > 0)

print("\nJUMLAH PIXEL PUTIH:")
print(f"Binary Original  : {hitung_pixel(binary)}")
print(f"Erosion          : {hitung_pixel(erosi)}")
print(f"Dilation         : {hitung_pixel(dilasi)}")
print(f"Opening          : {hitung_pixel(opening)}")
print(f"Closing          : {hitung_pixel(closing)}")

print("\nJUMLAH OBJEK TERDETEKSI:")
print(f"Binary Original  : {hitung_objek(binary)} objek")
print(f"Erosion          : {hitung_objek(erosi)} objek")
print(f"Dilation         : {hitung_objek(dilasi)} objek")
print(f"Opening          : {hitung_objek(opening)} objek")
print(f"Closing          : {hitung_objek(closing)} objek")

print("\nKESIMPULAN:")
print("1. EROSION: Menghilangkan noise, objek mengecil")
print("2. DILATION: Mengisi gap, objek membesar")
print("3. OPENING: Membersihkan noise luar, bentuk dipertahankan")
print("4. CLOSING: Mengisi lubang dalam, menggabungkan fragmen")
print("5. TERBAIK: CLOSING untuk deteksi kapsul (kontur lebih utuh)")
print("=" * 50)