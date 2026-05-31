import os
import re
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

from sklearn.metrics import precision_score, recall_score
import sys

# Añadimos 'src' al path para poder importar nuestros modulos locales
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from metrics import dice_coef, dice_loss, bce_dice_loss

img_dir = os.path.join("data", "test", "images")
gt1_dir = os.path.join("data", "test", "1st_manual")
gt2_dir = os.path.join("data", "test", "2nd_manual")
fov_dir = os.path.join("data", "test", "mask")
models_dir = os.path.join("models")
out_dir = os.path.join("predictions")

os.makedirs(out_dir, exist_ok=True)

patch_size = (128, 128)
input_shape = (128, 128, 3)

def leer_img(ruta):
    img = cv2.imread(ruta, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Error al leer la imagen: {ruta}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Aplicar filtro CLAHE al canal L
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    
    return img.astype(np.float32) / 255.0

def leer_mascara(ruta):
    mascara = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if mascara is None:
        return None
    return (mascara > 127).astype(np.float32)

def obtener_numero(nombre):
    num = re.match(r"(\d+)", nombre)
    return num.group(1) if num else nombre

def predecir_parches(modelo, img, p_size=(128, 128), stride=64):
    h, w, _ = img.shape
    ph, pw = p_size

    acumulador = np.zeros((h, w), dtype=np.float32)
    contador = np.zeros((h, w), dtype=np.float32)

    filas = list(range(0, h - ph + 1, stride))
    cols = list(range(0, w - pw + 1, stride))

    if not filas: filas = [0]
    if not cols: cols = [0]
    if filas[-1] + ph < h: filas.append(h - ph)
    if cols[-1] + pw < w: cols.append(w - pw)

    parches = []
    coords = []
    
    for r in filas:
        for c in cols:
            parches.append(img[r:r+ph, c:c+pw, :])
            coords.append((r, c))

    parches = np.array(parches, dtype=np.float32)
    preds = modelo.predict(parches, verbose=0, batch_size=16)

    for i, (r, c) in enumerate(coords):
        acumulador[r:r+ph, c:c+pw] += preds[i, :, :, 0]
        contador[r:r+ph, c:c+pw] += 1.0

    contador[contador == 0] = 1.0
    return acumulador / contador

def cargar_modelos(carpeta):
    rutas = sorted(glob.glob(os.path.join(carpeta, "fold_*.keras")))
    if not rutas:
        raise FileNotFoundError("No se han encontrado modelos. Ejecuta train.py primero.")
    
    modelos = []
    for r in rutas:
        print(f"Cargando modelo: {r}")
        m = keras.models.load_model(r, custom_objects={"dice_coef": dice_coef, "dice_loss": dice_loss})
        modelos.append(m)
        
    return modelos

def ensemble_predict(modelos, img, p_size=(128, 128), stride=64):
    suma = np.zeros(img.shape[:2], dtype=np.float32)
    
    img_h = cv2.flip(img, 1)
    img_v = cv2.flip(img, 0)
    
    for m in modelos:
        suma += predecir_parches(m, img, p_size, stride)
        
        p_h = predecir_parches(m, img_h, p_size, stride)
        suma += cv2.flip(p_h, 1)
        
        p_v = predecir_parches(m, img_v, p_size, stride)
        suma += cv2.flip(p_v, 0)
        
    return suma / (len(modelos) * 3.0)

def obtener_rutas_test():
    exts = ["*.tif", "*.png", "*.jpg", "*.bmp", "*.gif"]
    imgs, gt1s, gt2s, fovs = [], [], [], []

    for ext in exts:
        # Aquí está la corrección: usar os.path.join para cada búsqueda
        imgs += glob.glob(os.path.join(img_dir, ext))
        gt1s += glob.glob(os.path.join(gt1_dir, ext))
        gt2s += glob.glob(os.path.join(gt2_dir, ext))
        fovs += glob.glob(os.path.join(fov_dir, ext))

    idx_imgs = {obtener_numero(os.path.basename(r)): r for r in imgs}
    idx_gt1 = {obtener_numero(os.path.basename(r)): r for r in gt1s}
    idx_gt2 = {obtener_numero(os.path.basename(r)): r for r in gt2s}
    idx_fov = {obtener_numero(os.path.basename(r)): r for r in fovs}

    comunes = sorted(set(idx_imgs) & set(idx_gt1) & set(idx_gt2))
    
    if not comunes:
        raise FileNotFoundError("No se encontraron coincidencias de archivos en las carpetas de test.")

    imgs_fin = [idx_imgs[n] for n in comunes]
    gt1_fin = [idx_gt1[n] for n in comunes]
    gt2_fin = [idx_gt2[n] for n in comunes]
    fov_fin = [idx_fov.get(n) for n in comunes]

    print(f"Encontradas {len(comunes)} imágenes para evaluar.")
    return imgs_fin, gt1_fin, gt2_fin, fov_fin

def calcular_dice(pred, mask):
    inter = np.sum(pred * mask)
    denominador = np.sum(pred) + np.sum(mask)
    if denominador == 0:
        return 1.0
    return (2.0 * inter + 1e-6) / (denominador + 1e-6)

def run_evaluation():
    modelos = cargar_modelos(models_dir)
    imgs, gt1s, gt2s, fovs = obtener_rutas_test()

    res_dice_prom = []
    res_dice1, res_dice2 = [], []
    res_precision, res_recall = [], []

    for i, (r_img, r_gt1, r_gt2, r_fov) in enumerate(zip(imgs, gt1s, gt2s, fovs), 1):
        nombre = os.path.basename(r_img).split('.')[0]
        print(f"\n[{i}/{len(imgs)}] Evaluando: {nombre}")

        img = leer_img(r_img)
        gt1 = leer_mascara(r_gt1)
        gt2 = leer_mascara(r_gt2)
        fov = leer_mascara(r_fov) if r_fov else None

        prob = ensemble_predict(modelos, img, patch_size, 64)

        prob_uint8 = (prob * 255).astype(np.uint8)
        
        
        umbral_otsu_val, _ = cv2.threshold(prob_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        umbral_real = umbral_otsu_val / 255.0
        
        umbral_ajustado = max(0, umbral_real - 0.040)
        _, pred_uint8 = cv2.threshold(prob_uint8, int(umbral_ajustado * 255), 255, cv2.THRESH_BINARY)
        print(f"    -> Umbral Otsu: {umbral_real:.4f} | Ajustado: {umbral_ajustado:.4f}")
        
        pred = (pred_uint8 / 255.0).astype(np.float32)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        pred_uint8 = cv2.morphologyEx(pred_uint8, cv2.MORPH_CLOSE, kernel, iterations=1)
        pred = (pred_uint8 / 255.0).astype(np.float32)

        # Filtro de limpiado
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_uint8, connectivity=8)
        pred_limpia = np.zeros_like(pred)
        for j in range(1, num_labels):
            if stats[j, cv2.CC_STAT_AREA] >= 2:  
                pred_limpia[labels == j] = 1.0
        pred = pred_limpia

        if fov is not None:
            if fov.shape != pred.shape:
                fov = cv2.resize(fov, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred *= fov
            gt1 *= fov
            gt2 *= fov

        d1 = calcular_dice(pred, gt1)
        d2 = calcular_dice(pred, gt2)
        d_prom = (d1 + d2) / 2.0
        
        res_dice1.append(d1)
        res_dice2.append(d2)
        res_dice_prom.append(d_prom)

        gt_union = np.maximum(gt1, gt2)
        gt_plano = gt_union.flatten()
        pred_plano = pred.flatten()
            
        prec = precision_score(gt_plano, pred_plano, zero_division=0)
        rec = recall_score(gt_plano, pred_plano, zero_division=0)
        
        res_precision.append(prec)
        res_recall.append(rec)

        print(f"  Exp1: {d1:.4f} | Exp2: {d2:.4f} | Media: {d_prom:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

        cv2.imwrite(os.path.join(out_dir, f"{nombre}_pred.png"), (pred * 255).astype(np.uint8))

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(f"{nombre} - DICE Media: {d_prom:.4f}")
        
        axes[0, 0].imshow(img)
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")
        
        axes[0, 1].imshow(gt1, cmap="gray")
        axes[0, 1].set_title("Experto 1")
        axes[0, 1].axis("off")
        
        axes[1, 0].imshow(gt2, cmap="gray")
        axes[1, 0].set_title("Experto 2")
        axes[1, 0].axis("off")
        
        axes[1, 1].imshow(pred, cmap="gray")
        axes[1, 1].set_title(f"Predicción (umbral {umbral_real:.4f})")
        axes[1, 1].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{nombre}_comparacion.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    print("\n--- RESULTADOS FINALES ---")
    nombres = [os.path.basename(r).split('.')[0] for r in imgs]
    
    for nom, d1, d2, d_prom in zip(nombres, res_dice1, res_dice2, res_dice_prom):
        estado = "OK" if d_prom >= 0.75 else "FAIL"
        print(f"[{estado}] {nom} -> DICE Exp1: {d1:.4f} | DICE Exp2: {d2:.4f} | DICE Media: {d_prom:.4f}")
        
    media_total = np.mean(res_dice_prom)
    aprobados = sum(1 for d in res_dice_prom if d >= 0.75)
    
    print("\nResumen general:")
    print(f"  DICE Experto 1 : {np.mean(res_dice1):.4f}")
    print(f"  DICE Experto 2 : {np.mean(res_dice2):.4f}")
    print(f"  DICE Media     : {media_total:.4f}")
    print(f"  Precisión      : {np.mean(res_precision):.4f}")
    print(f"  Recall         : {np.mean(res_recall):.4f}")
    print(f"  Aprobados      : {aprobados}/{len(res_dice_prom)}")
    
    if media_total >= 0.75:
        print("\n¡Umbral de 0.75 superado!")
    else:
        print(f"\nFaltan {0.75 - media_total:.4f} puntos para superar el umbral.")


if __name__ == "__main__":
    run_evaluation()