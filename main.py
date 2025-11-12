# -*- coding: utf-8 -*-
import os, glob, json, hashlib, concurrent.futures
import cv2
import numpy as np
from pillow_heif import register_heif_opener
from PIL import Image
register_heif_opener()

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, filedialog, Button, Label, messagebox
from pathlib import Path

DB_PATH = "color_db.npz"

# Ruta a tus plantillas
BASE_DIR = Path(__file__).resolve().parent
TRAIN_ROOT = (BASE_DIR / "base de datos - billetes" / "billetes").resolve() 
AUTO_UPDATE_ON_START = True

print("Ruta de plantillas para auto-actualizar:", TRAIN_ROOT)        

# Velocidad de entrenamiento
TRAIN_RESIZE_LONG = 900             
MAX_WORKERS = max(2, min(8, os.cpu_count() or 4))

# Parámetros del descriptor
BINS = 36
SAT_MIN = 30
VAL_MIN = 40
WB_P = 6

# ========================= Utilidades de carga ========================= #
def cargar_imagen(ruta):
    if ruta.lower().endswith((".heic", ".heif")):
        imagen = Image.open(ruta)
        return cv2.cvtColor(np.array(imagen), cv2.COLOR_RGB2BGR)
    imagen = cv2.imread(ruta)
    if imagen is None:
        raise ValueError(f"No se pudo cargar la imagen: {ruta}")
    return imagen

def resize_max(im_bgr, long_max=TRAIN_RESIZE_LONG):
    H, W = im_bgr.shape[:2]
    L = max(H, W)
    if L > long_max:
        s = long_max / float(L)
        im_bgr = cv2.resize(im_bgr, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)
    return im_bgr

# ====================== Balance de blancos ====================== #
def white_balance_shades_of_gray(img_bgr, p=WB_P):
    eps = 1e-6
    img = img_bgr.astype(np.float32)
    r = np.power(np.mean(np.power(img[:, :, 2], p)), 1.0 / p) + eps
    g = np.power(np.mean(np.power(img[:, :, 1], p)), 1.0 / p) + eps
    b = np.power(np.mean(np.power(img[:, :, 0], p)), 1.0 / p) + eps
    k = (r + g + b) / 3.0
    gain = np.array([k / b, k / g, k / r], dtype=np.float32)
    out = img * gain
    return np.clip(out, 0, 255).astype(np.uint8)

# ======================= Pipeline ======================== #
def procesar_imagen(imagen_bgr):
    g = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.equalizeHist(g)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    return g / 255.0

# ============== Suavizado circular para histograma Hue ================ #
def smooth_circular_hist(h, k=(1, 2, 3, 2, 1)):
    h = np.asarray(h, dtype=np.float32).ravel()
    k = np.asarray(k, dtype=np.float32); k = k / (k.sum() + 1e-9)
    h_pad = np.concatenate([h[-2:], h, h[:2]], axis=0)
    hs = np.convolve(h_pad, k, mode='same')[2:-2].astype(np.float32)
    hs = hs / (hs.sum() + 1e-9)
    return hs.reshape((-1, 1))

# ========================== Color (HSV + Hue) ========================== #
COLOR_DB = {}        
META_DB  = {"classes":{}}  

def cargar_db_color(path=DB_PATH):
    global COLOR_DB, META_DB
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        etiquetas = list(data["labels"])
        hists = list(data["hists"])
        COLOR_DB = {etq: h.astype(np.float32) for etq, h in zip(etiquetas, hists)}
        if "meta_json" in data:
            try:
                META_DB = json.loads(str(data["meta_json"]))
            except Exception:
                META_DB = {"classes":{}}
        else:
            META_DB = {"classes":{}}
    else:
        COLOR_DB = {}
        META_DB  = {"classes":{}}

def guardar_db_color(path=DB_PATH):
    labels = np.array(list(COLOR_DB.keys()), dtype=object)
    hists  = np.array(list(COLOR_DB.values()), dtype=object)
    meta_json = json.dumps(META_DB)
    np.savez(path, labels=labels, hists=hists, meta_json=meta_json)

def hue_histogram(im_bgr, bins=BINS, sat_min=SAT_MIN, val_min=VAL_MIN, return_mask=True):
    hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((s >= sat_min) & (v >= val_min)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    hist = cv2.calcHist([h], [0], mask, [bins], [0, 180]).astype(np.float32)
    hist /= (hist.sum() + 1e-9)
    hist = smooth_circular_hist(hist)
    edges = np.linspace(0, 180, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if return_mask:
        return hist, centers, mask
    return hist, centers

def dominante_hue(hist, centers):
    return float(centers[int(np.argmax(hist))])

def clasificar_por_plantillas(hist, min_conf=0.55, min_gap=0.10):
    if not COLOR_DB:
        return None, 0.0
    hist = smooth_circular_hist(hist)
    candidatos = []
    for etq, templ in COLOR_DB.items():
        t = smooth_circular_hist(templ)
        corr = cv2.compareHist(hist, t, cv2.HISTCMP_CORREL)
        bhat = cv2.compareHist(hist, t, cv2.HISTCMP_BHATTACHARYYA)
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

# --------- Reglas basadas en hue dominante ------- #
RANGOS_HUE = {
    "Billete de 2":   [(16, 30)],
    "Billete de 5":   [(0, 15), (165, 180)],
    "Billete de 20":  [(30, 50)],
    "Billete de 100": [(60, 85)],
}
def clasificar_por_reglas(hue_dom):
    for etiqueta, rangos in RANGOS_HUE.items():
        for (a, b) in rangos:
            if a <= hue_dom <= b:
                return etiqueta
    return None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif", ".tif", ".tiff", ".jfif", ".webp"}

def _files_list_signature(dir_path):
    """
    Busca imágenes **recursivamente** dentro de la clase.
    Firma = ruta_rel + tamaño + mtime (ordena para que sea estable).
    """
    dir_path = Path(dir_path)
    entries = []
    for root, _, files in os.walk(dir_path):
        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext in IMG_EXTS:
                p = Path(root) / fname
                try:
                    st = p.stat()
                    rel = p.relative_to(dir_path).as_posix()
                    entries.append((rel, int(st.st_size), int(st.st_mtime)))
                except FileNotFoundError:
                    pass

    entries.sort()
    h = hashlib.sha1()
    for rel, sz, mt in entries:
        h.update(rel.encode("utf-8"))
        h.update(sz.to_bytes(8, "little", signed=False))
        h.update(mt.to_bytes(8, "little", signed=False))

    abs_paths = [str(dir_path / rel) for rel, _, _ in entries]
    return abs_paths, h.hexdigest()

def _hist_de_ruta(path, bins, sat_min, val_min):
    im = cargar_imagen(path)
    im = resize_max(im, TRAIN_RESIZE_LONG)
    im = white_balance_shades_of_gray(im, p=WB_P)
    hist, _, _ = hue_histogram(im, bins=bins, sat_min=sat_min, val_min=val_min, return_mask=True)
    return hist

# =================== Auto-actualizar DB desde carpeta =================== #
def auto_update_db_from_folder(root):
    """
    Usa TRAIN_ROOT y solo recalcula las clases que cambiaron (o si cambiaste parámetros).
    Sin diálogos; imprime estado en consola y devuelve un mensaje corto.
    """
    global COLOR_DB, META_DB
    # Cargar DB existente
    if not COLOR_DB or not META_DB:
        cargar_db_color(DB_PATH)

    if not os.path.isdir(root):
        msg = f"[AUTO] Carpeta de plantillas no encontrada: {root}"
        print(msg)
        return msg

    # Parámetros actuales
    params_actuales = {"bins": BINS, "sat_min": SAT_MIN, "val_min": VAL_MIN, "wb_p": WB_P, "resize": TRAIN_RESIZE_LONG}
    meta_classes = META_DB.get("classes", {})
    meta_params  = {k: META_DB.get(k) for k in ("bins","sat_min","val_min","wb_p","resize")}
    mismos_params = (meta_params == params_actuales)

    clases = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not clases:
        msg = "[AUTO] No hay subcarpetas de clases en TRAIN_ROOT."
        print(msg)
        return msg

    clases_a_recalcular = []
    nuevas_plantillas = {}
    total_imgs = 0

    for clase in clases:
        carpeta = os.path.join(root, clase)
        paths, fp = _files_list_signature(carpeta)

        meta_clase = meta_classes.get(clase)
        if meta_clase and meta_clase.get("fingerprint") == fp and mismos_params and clase in COLOR_DB:
            continue  # sin cambios

        clases_a_recalcular.append((clase, carpeta, paths, fp))

    clases_existentes = set(clases)
    clases_en_db = set(COLOR_DB.keys())
    clases_sobrantes = list(clases_en_db - clases_existentes)

    if not clases_a_recalcular and not clases_sobrantes:
        msg = "[AUTO] DB ya actualizada (sin cambios)."
        print(msg)
        return msg

    # Recalcular necesarias (paralelo)
    for clase, carpeta, paths, fp in clases_a_recalcular:
        if not paths:
            continue
        hists = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(_hist_de_ruta, p, BINS, SAT_MIN, VAL_MIN) for p in paths]
            for f in concurrent.futures.as_completed(futs):
                try:
                    hists.append(f.result())
                except Exception as e:
                    print(f"[WARN] No se pudo procesar {carpeta}: {e}")
        if hists:
            H = np.stack(hists, axis=0).astype(np.float32)
            H = H / (H.sum(axis=1, keepdims=True) + 1e-9)
            templ = H.mean(axis=0).astype(np.float32)
            templ = templ / (templ.sum() + 1e-9)
            nuevas_plantillas[clase] = templ
            total_imgs += len(hists)
            meta_classes[clase] = {"fingerprint": fp, "count": len(paths)}

    # Aplicar cambios
    COLOR_DB.update(nuevas_plantillas)
    for cls_del in clases_sobrantes:
        COLOR_DB.pop(cls_del, None)
        meta_classes.pop(cls_del, None)

    META_DB["classes"] = meta_classes
    META_DB.update(params_actuales)
    guardar_db_color(DB_PATH)

    msg = f"[AUTO] Recalc: {len(nuevas_plantillas)} | Del: {len(clases_sobrantes)} | Imgs: {total_imgs} | Clases totales: {len(COLOR_DB)}"
    print(msg)
    return msg

# ================== Interfaz y flujo de visualización ================== #
_last_image = None
lbl_resultados = None
lbl_pred = None  # etiqueta grande arriba
lbl_status = None

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

    # --- Resize para display ---
    h, w = imagen.shape[:2]
    imagen_resized = cv2.resize(imagen, (max(1, w // 2), max(1, h // 2)))

    # --- WB antes de HSV ---
    imagen_resized = white_balance_shades_of_gray(imagen_resized, p=WB_P)

    # --- Métricas/visualización ---
    imagen_pre = procesar_imagen(imagen_resized)

    # --- HSV: hist + máscara + clasificación ---
    hist_h, centers, sv_mask = hue_histogram(
        imagen_resized, bins=META_DB.get("bins", BINS),
        sat_min=META_DB.get("sat_min", SAT_MIN),
        val_min=META_DB.get("val_min", VAL_MIN), return_mask=True
    )
    hue_dom = dominante_hue(hist_h, centers)

    etiqueta_color, conf = clasificar_por_plantillas(hist_h, min_conf=0.55, min_gap=0.10)
    modo = "plantillas"
    if etiqueta_color is None:
        etiqueta_color = clasificar_por_reglas(hue_dom)
        modo = "reglas"
        conf = 0.0 if etiqueta_color is None else conf

    # --- Métricas de intensidad ---
    media_hist = float(np.mean(imagen_pre.ravel()))
    mu = float(np.mean(imagen_pre))
    sigma = float(np.std(imagen_pre))
    snr = float(mu / sigma) if sigma != 0 else 0.0

    # --- Dibujo (2x2) ---
    axs[0,0].cla()
    axs[0,0].imshow(cv2.cvtColor(imagen_resized, cv2.COLOR_BGR2RGB))
    axs[0,0].set_title("Imagen Original (WB)")
    axs[0,0].axis("off")

    # Overlay del resultado sobre la imagen
    if etiqueta_color:
        texto_overlay = f"{etiqueta_color}  " + (f"(conf {conf:.2f})" if modo=="plantillas" else "(reglas)")
        axs[0,0].text(
            0.02, 0.04, texto_overlay,
            transform=axs[0,0].transAxes,
            fontsize=14, color="w",
            bbox=dict(facecolor="black", alpha=0.55, pad=4, edgecolor="none")
        )

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

    # --- Etiquetas (más visible) ---
    if etiqueta_color:
        if modo == "plantillas":
            texto_color = f"BILLETE: {etiqueta_color}  |  Confianza: {conf:.2f}"
            color_fg = "#16a34a" if conf >= 0.75 else "#ca8a04"  # verde/ámbar
        else:
            texto_color = f"BILLETE: {etiqueta_color}  |  (Reglas por Hue)"
            color_fg = "#2563eb"  # azul
    else:
        texto_color = "SIN DECISIÓN"
        color_fg = "#dc2626"  # rojo

    lbl_pred.config(text=texto_color, fg=color_fg)  # grande arriba

    texto_small = f"Media gris: {media_hist:.3f} | SNR: {snr:.3f}"
    lbl_resultados.config(text=texto_small)

    canvas.draw()

# =============================== Main =============================== #
def main():
    global fig, axs, canvas, lbl_resultados, lbl_pred, lbl_status
    cargar_db_color(DB_PATH)

    # Auto-actualizar DB desde carpeta fija (si está activado)
    status_msg = ""
    if AUTO_UPDATE_ON_START:
        status_msg = auto_update_db_from_folder(TRAIN_ROOT)

    root = Tk()
    root.title("Lector de Billetes - Color HSV")
    root.geometry("1050x800")

    # Etiqueta grande para resultado
    lbl_pred = Label(root, text="BILLETE: —", font=("Segoe UI", 20, "bold"))
    lbl_pred.pack(pady=6)

    # Status pequeño de la DB
    lbl_status = Label(root, text=status_msg or f"Clases en DB: {len(COLOR_DB)}", font=("Segoe UI", 9))
    lbl_status.pack(pady=2)

    # Botón para cargar imagen
    Button(root, text="Cargar Imagen", command=seleccionar_imagen).pack(pady=8)

    # Canvas Matplotlib
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.2))
    fig.subplots_adjust(hspace=0.28, wspace=0.16)
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    # Etiqueta secundaria (métricas)
    lbl_resultados = Label(root, text="—", font=("Segoe UI", 11))
    lbl_resultados.pack(pady=6)

    root.mainloop()

if __name__ == "__main__":
    main()
