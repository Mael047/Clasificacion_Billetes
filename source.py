import cv2 
import numpy as np
from pillow_heif import register_heif_opener
from PIL import Image
register_heif_opener()

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, filedialog, Button, Label

# ---------------- Procesamiento ---------------- #
def procesar_imagen(imagen):
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imagen = cv2.equalizeHist(imagen)
    imagen = cv2.GaussianBlur(imagen, (5, 5), 0)
    return imagen/255.0

def detector_bordes(imagen):
    img_uint8 = (imagen * 255).astype(np.uint8)
    sobelx = cv2.Sobel(img_uint8, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img_uint8, cv2.CV_64F, 0, 1, ksize=5)
    bordes = np.sqrt(sobelx**2 + sobely**2)
    return bordes / bordes.max()

def transformada_furier(imagen):
    f = np.fft.fft2(imagen)
    f_transformada = np.fft.fftshift(f)
    magnitud = np.abs(f_transformada)
    return np.log1p(magnitud)

def cargar_imagen(ruta):
    if ruta.lower().endswith(".heic"):
        imagen = Image.open(ruta)
        return cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)
    else:
        imagen = cv2.imread(ruta)
        if imagen is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta}")
        return imagen

# ---------------- Interfaz ---------------- #
def seleccionar_imagen():
    ruta = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.heic *.tif *.tiff")]
    )
    if not ruta:
        return

    # Cargar imagen
    imagen = cargar_imagen(ruta)
    alto, ancho = imagen.shape[:2]
    imagen_resized = cv2.resize(imagen, (ancho//2, alto//2))

    # Procesar
    imagen_pre = procesar_imagen(imagen_resized)
    fourier = transformada_furier(imagen_pre)
    bordes = detector_bordes(imagen_pre)

    axs[0,0].cla()
    axs[0,0].imshow(cv2.cvtColor(imagen_resized, cv2.COLOR_BGR2RGB))
    axs[0,0].set_title("Imagen Original")
    axs[0,0].axis("off")

    axs[0,1].cla()
    axs[0,1].imshow(imagen_pre, cmap="gray")
    axs[0,1].set_title("Preprocesada")
    axs[0,1].axis("off")

    axs[0,2].cla()
    axs[0,2].hist(imagen_pre.ravel(), bins=256, color="black")
    axs[0,2].set_title("Histograma")

    axs[1,0].cla()
    axs[1,0].imshow(fourier, cmap="gray")
    axs[1,0].set_title("Fourier")
    axs[1,0].axis("off")

    axs[1,1].cla()
    axs[1,1].imshow(bordes, cmap="gray")
    axs[1,1].set_title("Bordes Sobel")
    axs[1,1].axis("off")

    # Métricas
    media_hist = np.mean(imagen_pre.ravel())
    mu = np.mean(imagen_pre)
    sigma = np.std(imagen_pre)
    snr = mu / sigma if sigma != 0 else 0

    lbl_resultados.config(text=f"Media hist: {media_hist:.3f} | SNR: {snr:.3f}")
    categoria = categorizar_billete(media_hist, snr)
    if categoria:
        lbl_resultados.config(text=lbl_resultados.cget("text") + f" | Categoría: {categoria}")
    canvas.draw()

def categorizar_billete(media_hist, snr):
    if media_hist > 0.510 and media_hist < 0.535 and snr > 1.749 and snr < 1.870:
        return "Billete de 100"
    elif media_hist > 0.505 and media_hist < 0.510 and snr > 1.755 and snr < 1.815:
        return "Billete de 5"
    elif media_hist > 0.490 and media_hist < 0.505 and snr > 1.820 and snr < 1.870:
        return "Billete de 20"
    elif media_hist >= 0.504 and media_hist < 0.510 and snr > 1.765 and snr < 1.856:
        return "Billete de 2"
    
def main():
    global fig, axs, canvas

    root = Tk()
    root.title("Lector de Billetes - Procesamiento")
    root.geometry("900x700")

    lbl = Label(root, text="Seleccione una imagen para procesar:")
    lbl.pack(pady=10)

    btn = Button(root, text="Cargar Imagen", command=seleccionar_imagen)
    btn.pack(pady=5)

    # Crear figura una sola vez
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True)
    global lbl_resultados
    lbl_resultados = Label(root, text="Resultados: ")
    lbl_resultados.pack(pady=5)

    root.mainloop()


if __name__ == "__main__":
    main()
