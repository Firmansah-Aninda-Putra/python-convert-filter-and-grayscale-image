import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca gambar dari direktori yang ditentukan
img = cv2.imread(r"C:\Users\bumii\Desktop\GRAYSCALE IMAGE\saya.jpeg")

# Mengecek apakah gambar berhasil dibaca
if img is None:
    print("Error: Tidak dapat membaca gambar. Periksa path file!")
    exit()

# Konversi ke grayscale
img_gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Mendapatkan dimensi asli gambar
height, width = img_gs.shape

# === FILTER 1: BLUR (Average Filter) ===
# Menggunakan kernel 5x5 untuk smoothing
blur = cv2.blur(img_gs, (5, 5))

# === FILTER 2: GAUSSIAN BLUR ===
# Gaussian blur dengan kernel 5x5 dan sigma=0 (otomatis dihitung)
gaussian_blur = cv2.GaussianBlur(img_gs, (5, 5), 0)

# === FILTER 3: MEDIAN BLUR ===
# Median filter dengan kernel size 5 (sangat baik untuk menghilangkan noise salt & pepper)
median = cv2.medianBlur(img_gs, 5)

# === FILTER 4: BILATERAL BLUR ===
# Bilateral filter menjaga edge sambil smoothing (d=9, sigmaColor=75, sigmaSpace=75)
bilateral_blur = cv2.bilateralFilter(img_gs, 9, 75, 75)

# === FILTER 5: SHARPENING ===
# Menggunakan metode unsharp masking: sharpened = original + (original - blurred)
# Rumus: sharp = 2*original - 1*blurred
sharp1 = cv2.addWeighted(img_gs, 2, blur, -1, 0)

# Alternatif sharpening menggunakan kernel
kernel_sharp = np.array([[-1, -1, -1],
                         [-1,  9, -1],
                         [-1, -1, -1]])
sharp2 = cv2.filter2D(img_gs, -1, kernel_sharp)

# === MENAMPILKAN HASIL ===
# Fungsi untuk menampilkan gambar dengan ukuran yang konsisten
def show_image(window_name, image, x_pos, y_pos):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 400, 500)
    cv2.moveWindow(window_name, x_pos, y_pos)
    cv2.imshow(window_name, image)

# Menampilkan semua hasil dengan posisi teratur
show_image("1. Original Grayscale", img_gs, 0, 0)
show_image("2. Blur Filter", blur, 420, 0)
show_image("3. Gaussian Blur", gaussian_blur, 840, 0)
show_image("4. Median Blur", median, 0, 550)
show_image("5. Bilateral Blur", bilateral_blur, 420, 550)
show_image("6. Sharpening (Unsharp Mask)", sharp1, 840, 550)

# === MENAMPILKAN HISTOGRAM ===
# Membuat plot histogram untuk perbandingan
plt.figure(figsize=(15, 10))

# Histogram Original
plt.subplot(2, 3, 1)
hist_original = cv2.calcHist([img_gs], [0], None, [256], [0, 256])
plt.plot(hist_original, color='gray')
plt.title("Histogram: Original Grayscale")
plt.xlabel("Intensity Value")
plt.ylabel("Pixel Count")
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)

# Histogram Blur
plt.subplot(2, 3, 2)
hist_blur = cv2.calcHist([blur], [0], None, [256], [0, 256])
plt.plot(hist_blur, color='blue')
plt.title("Histogram: Blur Filter")
plt.xlabel("Intensity Value")
plt.ylabel("Pixel Count")
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)

# Histogram Gaussian Blur
plt.subplot(2, 3, 3)
hist_gaussian = cv2.calcHist([gaussian_blur], [0], None, [256], [0, 256])
plt.plot(hist_gaussian, color='green')
plt.title("Histogram: Gaussian Blur")
plt.xlabel("Intensity Value")
plt.ylabel("Pixel Count")
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)

# Histogram Median
plt.subplot(2, 3, 4)
hist_median = cv2.calcHist([median], [0], None, [256], [0, 256])
plt.plot(hist_median, color='orange')
plt.title("Histogram: Median Blur")
plt.xlabel("Intensity Value")
plt.ylabel("Pixel Count")
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)

# Histogram Bilateral
plt.subplot(2, 3, 5)
hist_bilateral = cv2.calcHist([bilateral_blur], [0], None, [256], [0, 256])
plt.plot(hist_bilateral, color='red')
plt.title("Histogram: Bilateral Blur")
plt.xlabel("Intensity Value")
plt.ylabel("Pixel Count")
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)

# Histogram Sharpening
plt.subplot(2, 3, 6)
hist_sharp = cv2.calcHist([sharp1], [0], None, [256], [0, 256])
plt.plot(hist_sharp, color='purple')
plt.title("Histogram: Sharpening")
plt.xlabel("Intensity Value")
plt.ylabel("Pixel Count")
plt.xlim([0, 256])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# === MENYIMPAN HASIL ===
# Menyimpan semua hasil ke file
cv2.imwrite(r"C:\Users\bumii\Desktop\GRAYSCALE IMAGE\hasil_grayscale.jpg", img_gs)
cv2.imwrite(r"C:\Users\bumii\Desktop\GRAYSCALE IMAGE\hasil_blur.jpg", blur)
cv2.imwrite(r"C:\Users\bumii\Desktop\GRAYSCALE IMAGE\hasil_gaussian.jpg", gaussian_blur)
cv2.imwrite(r"C:\Users\bumii\Desktop\GRAYSCALE IMAGE\hasil_median.jpg", median)
cv2.imwrite(r"C:\Users\bumii\Desktop\GRAYSCALE IMAGE\hasil_bilateral.jpg", bilateral_blur)
cv2.imwrite(r"C:\Users\bumii\Desktop\GRAYSCALE IMAGE\hasil_sharpening.jpg", sharp1)

print("=" * 60)
print("PROGRAM FILTER CITRA DIGITAL")
print("=" * 60)
print(f"Dimensi gambar: {width} x {height} pixels")
print("\nSemua filter berhasil diaplikasikan!")
print("Hasil telah disimpan di folder yang sama dengan gambar asli.")
print("\nTekan sembarang tombol pada window gambar untuk keluar...")
print("=" * 60)

# Menunggu input keyboard
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("PENJELASAN MASING-MASING FILTER:")
print("=" * 60)