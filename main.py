# -*- coding: utf-8 -*-
import os, glob
import cv2
import numpy as np
from pillow_heif import register_heif_opener
from PIL import Image
register_heif_opener()

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, filedialog, Button, Label, messagebox

# ========================= Utilidades de carga ========================= #
def cargar_imagen(ruta):
    if ruta.lower().endswith((".heic", ".heif")):
        imagen = Image.open(ruta)
        return cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)
    else:
        imagen = cv2.imread(ruta)
        if imagen is None:
            raise ValueError(f"No se pudo cargar la imagen: {ruta}")
        return imagen

# ====================== Mejora: balance de blancos ===================== #
def white_balance_shades_of_gray(img_bgr, p=6):
    """
    Balance de blancos 'Shades-of-Gray' (Minkowski p-norm).
    Estabiliza el tono bajo diferentes iluminaciones.
    """
    eps = 1e-6
    img = img_bgr.astype(np.float32)
    r = np.power(np.mean(np.power(img[:, :, 2], p)), 1.0 / p) + eps
    g = np.power(np.mean(np.power(img[:, :, 1], p)), 1.0 / p) + eps
    b = np.power(np.mean(np.power(img[:, :, 0], p)), 1.0 / p) + eps
    k = (r + g + b) / 3.0
    gain = np.array([k / b, k / g, k / r], dtype=np.float32)
    out = img * gain
    return np.clip(out, 0, 255).astype(np.uint8)

# ======================= Pipeline (intensidad) ======================== #
def procesar_imagen(imagen_bgr):
    """
    Gris + equalize + blur para métricas/visualización.
    """
    g = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return g / 255.0

# ============== Suavizado circular para histograma Hue ================ #
def smooth_circular_hist(h, k=(1, 2, 3, 2, 1)):
    """
    Suaviza un histograma circular (0° ~ 180° vecinos).
    Mantiene L1-normalización.
    """
    h = np.asarray(h, dtype=np.float32).ravel()
    k = np.asarray(k, dtype=np.float32)
    k = k / (k.sum() + 1e-9)

    # Wrap-around: tomar 2 del final y 2 del inicio
    h_pad = np.concatenate([h[-2:], h, h[:2]], axis=0)
    hs = np.convolve(h_pad, k, mode='same')[2:-2].astype(np.float32)
    hs = hs / (hs.sum() + 1e-9)
    return hs.reshape((-1, 1))

# ========================== Color (HSV + Hue) ========================== #
COLOR_DB = {}
DB_PATH = "color_db.npz"

def cargar_db_color(path=DB_PATH):
    global COLOR_DB
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        etiquetas = list(data["labels"])
        hists = list(data["hists"])
        COLOR_DB = {etq: h.astype(np.float32) for etq, h in zip(etiquetas, hists)}
    else:
        COLOR_DB = {}

def guardar_db_color(path=DB_PATH):
    if COLOR_DB:
        labels = np.array(list(COLOR_DB.keys()), dtype=object)
        hists = np.array(list(COLOR_DB.values()), dtype=object)
        np.savez(path, labels=labels, hists=hists)

def hue_histogram(im_bgr, bins=36, sat_min=30, val_min=40, return_mask=True):
    """
    Histograma de Hue (0..180, OpenCV) con máscara por S/V y limpieza morfológica.
    Devuelve (hist L1-normalizado y suavizado, centers) y opcionalmente la máscara.
    """
    hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((s >= sat_min) & (v >= val_min)).astype(np.uint8) * 255

    # Limpieza de máscara: apertura pequeña
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    hist = cv2.calcHist([h], [0], mask, [bins], [0, 180]).astype(np.float32)
    hist /= (hist.sum() + 1e-9)
    hist = smooth_circular_hist(hist)  # suavizado circular

    edges = np.linspace(0, 180, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if return_mask:
        return hist, centers, mask
    return hist, centers

def dominante_hue(hist, centers):
    return float(centers[int(np.argmax(hist))])

def clasificar_por_plantillas(hist, min_conf=0.55, min_gap=0.10):
    """
    Compara contra COLOR_DB con correlación y Bhattacharyya.
    Aplica rechazo si la confianza < min_conf o si el gap top-2 < min_gap.
    Retorna (etiqueta, confianza). Si no hay DB o es dudoso, (None, best_conf).
    """
    if not COLOR_DB:
        return None, 0.0

    # Asegurar hist suavizado
    hist = smooth_circular_hist(hist)

    candidatos = []
    for etq, templ in COLOR_DB.items():
        t = smooth_circular_hist(templ)
        corr = cv2.compareHist(hist, t, cv2.HISTCMP_CORREL)                  # [-1,1] ↑ mejor
        bhat = cv2.compareHist(hist, t, cv2.HISTCMP_BHATTACHARYYA)           # [0,inf) ↓ mejor
        conf_corr = (corr + 1) / 2.0
        conf_bhat = float(np.exp(-bhat * 5.0))
        conf = float(0.7 * conf_corr + 0.3 * conf_bhat)
        candidatos.append((etq, conf))

    candidatos.sort(key=lambda x: x[1], reverse=True)
    best_label, best_conf = candidatos[0]
    gap = best_conf - (candidatos[1][1] if len(candidatos) > 1 else 0.0)

    if best_conf < min_conf or gap < min_gap:
        return None, best_conf
    return best_label, best_conf

# --------- Reglas (fallback) basadas en hue dominante (EDITABLES) ------- #
RANGOS_HUE = {
    "Billete de 2":   [(85, 120)],            # azules ~100
    "Billete de 5":   [(0, 15), (165, 180)],  # rojizos
    "Billete de 20":  [(12, 30)],             # naranjas
    "Billete de 100": [(60, 85)],             # verdosos
}

def clasificar_por_reglas(hue_dom):
    for etiqueta, rangos in RANGOS_HUE.items():
        for (a, b) in rangos:
            if a <= hue_dom <= b:
                return etiqueta
    return None

# =================== Entrenamiento desde carpetas ====================== #
def entrenar_desde_carpetas():
    """
    Carpeta raíz con subcarpetas por etiqueta; genera/actualiza color_db.npz.
    Aplica balance de blancos y la misma extracción de hist que en inferencia.
    """
    root = filedialog.askdirectory(title="Selecciona la carpeta raíz de plantillas")
    if not root:
        return
    clases = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not clases:
        messagebox.showerror("Error", "No se encontraron subcarpetas (clases).")
        return

    bins = 36
    db_local = {}

    for clase in clases:
        carpeta = os.path.join(root, clase)
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.heic", "*.heif", "*.tif", "*.tiff"):
            paths.extend(glob.glob(os.path.join(carpeta, ext)))
        if not paths:
            continue

        hists = []
        for p in paths:
            try:
                im = cargar_imagen(p)
                im = white_balance_shades_of_gray(im, p=6)
                hist, _cent, _mask = hue_histogram(im, bins=bins, sat_min=30, val_min=40, return_mask=True)
                hists.append(hist)
            except Exception as e:
                print(f"[WARN] No se pudo procesar {p}: {e}")

        if hists:
            H = np.stack(hists, axis=0).astype(np.float32)
            H = H / (H.sum(axis=1, keepdims=True) + 1e-9)
            templ = H.mean(axis=0).astype(np.float32)
            templ = templ / (templ.sum() + 1e-9)
            db_local[clase] = templ

    if not db_local:
        messagebox.showerror("Error", "No se pudieron crear plantillas (¿sin imágenes válidas?).")
        return

    COLOR_DB.update(db_local)
    guardar_db_color(DB_PATH)
    messagebox.showinfo("Listo", f"Plantillas creadas/actualizadas ({len(db_local)} clases).")
    if _last_image is not None:
        seleccionar_imagen(recargar=True)

# ================== Interfaz y flujo de visualización ================== #
_last_image = None  # para re-evaluar tras entrenar

def seleccionar_imagen(recargar=False):
    global _last_image

    if not recargar:
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.heic *.tif *.tiff")]
        )
        if not ruta:
            return
        imagen = cargar_imagen(ruta)
        _last_image = imagen.copy()
    else:
        if _last_image is None:
            return
        imagen = _last_image.copy()

    # --- Resize para display (mitad) ---
    h, w = imagen.shape[:2]
    imagen_resized = cv2.resize(imagen, (max(1, w // 2), max(1, h // 2)))

    # --- Mejora: balance de blancos antes de HSV ---
    imagen_resized = white_balance_shades_of_gray(imagen_resized, p=6)

    # --- Pipeline intensidad (para métricas/visualización) ---
    imagen_pre = procesar_imagen(imagen_resized)

    # --- HSV: hist de Hue + máscara S/V + clasificación ---
    hist_h, centers, sv_mask = hue_histogram(
        imagen_resized, bins=36, sat_min=30, val_min=40, return_mask=True
    )
    hue_dom = dominante_hue(hist_h, centers)

    etiqueta_color, conf = clasificar_por_plantillas(hist_h, min_conf=0.55, min_gap=0.10)
    if etiqueta_color is None:
        # Fallback por reglas (sin confianza cuantificada)
        etiqueta_color = clasificar_por_reglas(hue_dom)
        if etiqueta_color is None:
            conf = 0.0

    # --- Métricas de intensidad ---
    media_hist = float(np.mean(imagen_pre.ravel()))
    mu = float(np.mean(imagen_pre))
    sigma = float(np.std(imagen_pre))
    snr = float(mu / sigma) if sigma != 0 else 0.0

    # ===================== Dibujos (2x2) ===================== #
    axs[0,0].cla()
    axs[0,0].imshow(cv2.cvtColor(imagen_resized, cv2.COLOR_BGR2RGB))
    axs[0,0].set_title("Imagen Original (WB)")
    axs[0,0].axis("off")

    axs[0,1].cla()
    axs[0,1].imshow(imagen_pre, cmap="gray")
    axs[0,1].set_title("Preprocesada (gris)")
    axs[0,1].axis("off")

    axs[1,0].cla()
    axs[1,0].bar(centers, hist_h.ravel(), width=(centers[1] - centers[0]) * 0.9)
    axs[1,0].set_xlim(0, 180)
    axs[1,0].set_xlabel("Hue (° OpenCV 0–180)")
    axs[1,0].set_ylabel("Frecuencia")
    axs[1,0].set_title(f"Hist Hue • Hue dom: {hue_dom:.1f}°")

    axs[1,1].cla()
    axs[1,1].imshow(sv_mask, cmap="gray")
    axs[1,1].set_title("Máscara S/V (zonas válidas)")
    axs[1,1].axis("off")

    # ===================== Etiquetas ===================== #
    if etiqueta_color:
        if conf > 0:
            texto_color = f"{etiqueta_color} (conf: {conf:.2f})"
        else:
            texto_color = f"{etiqueta_color} (reglas)"
    else:
        texto_color = "Sin decisión"

    texto = f"Media gris: {media_hist:.3f} | SNR: {snr:.3f} | Billete de {texto_color}"
    lbl_resultados.config(text=texto)

    canvas.draw()

# =============================== Main =============================== #
def main():
    global fig, axs, canvas, lbl_resultados
    cargar_db_color(DB_PATH)

    root = Tk()
    root.title("Lector de Billetes - Color HSV con Plantillas (mejorado)")
    root.geometry("1000x750")

    Label(root, text="Cargar imagen o entrenar plantillas (color por HSV).").pack(pady=10)

    Button(root, text="Cargar Imagen", command=seleccionar_imagen).pack(pady=5)
    Button(root, text="Entrenar (carpetas)", command=entrenar_desde_carpetas).pack(pady=5)

    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.8))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    lbl_resultados = Label(root, text="Resultados: ")
    lbl_resultados.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()
