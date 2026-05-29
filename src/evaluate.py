# ===========================================================================
# evaluate.py — Evaluación sobre el conjunto de TEST del dataset DRIVE
# EJECUTAR DESDE LA RAIZ DEL PROYECTO: python evaluate.py
# ===========================================================================

import os
import re
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from metrics import dice_coef, dice_loss, bce_dice_loss


# ===========================================================================
# 0. CONFIGURACIÓN CENTRAL
# ===========================================================================

DIR_IMAGENES_TEST  = os.path.join("data", "test", "images")
DIR_MASCARAS_TEST  = os.path.join("data", "test", "1st_manual")
DIR_FOV_TEST       = os.path.join("data", "test", "mask")

DIR_MODELOS        = "modelos_guardados"
DIR_PREDICCIONES   = "predicciones"

PATCH_SIZE         = (128, 128)
UMBRAL             = 0.5
INPUT_SHAPE        = (128, 128, 3)
EXTENSIONES        = ["*.tif", "*.png", "*.jpg", "*.bmp", "*.gif"]

os.makedirs(DIR_PREDICCIONES, exist_ok=True)


# ===========================================================================
# 1. UTILIDADES DE LECTURA DE IMAGEN
# ===========================================================================

def leer_imagen_rgb(ruta: str) -> np.ndarray:
    img = cv2.imread(ruta, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def leer_mascara(ruta: str) -> np.ndarray:
    mask = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"No se pudo leer la mascara: {ruta}")
    return (mask > 127).astype(np.float32)


def leer_fov(ruta: str) -> np.ndarray:
    fov = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if fov is None:
        return None
    return (fov > 127).astype(np.float32)


def extraer_numero(nombre_base: str) -> str:
    """Extrae el número inicial del nombre de fichero. Ej: '01_test' -> '01'"""
    m = re.match(r"(\d+)", nombre_base)
    return m.group(1) if m else nombre_base


# ===========================================================================
# 2. PREDICCIÓN POR PARCHES SOLAPADOS
# ===========================================================================

def predecir_imagen_por_parches(modelo, imagen, patch_size=(128, 128), stride=64):
    H, W, C = imagen.shape
    ph, pw = patch_size

    acumulador = np.zeros((H, W), dtype=np.float32)
    contador   = np.zeros((H, W), dtype=np.float32)

    filas = list(range(0, H - ph + 1, stride))
    cols  = list(range(0, W - pw + 1, stride))

    if len(filas) == 0: filas = [0]
    if len(cols)  == 0: cols  = [0]
    if filas[-1] + ph < H: filas.append(H - ph)
    if cols[-1]  + pw < W: cols.append(W - pw)

    parches, posiciones = [], []
    for r in filas:
        for c in cols:
            parches.append(imagen[r:r+ph, c:c+pw, :])
            posiciones.append((r, c))

    parches_np   = np.array(parches, dtype=np.float32)
    predicciones = modelo.predict(parches_np, verbose=0, batch_size=16)

    for idx, (r, c) in enumerate(posiciones):
        acumulador[r:r+ph, c:c+pw] += predicciones[idx, :, :, 0]
        contador[r:r+ph, c:c+pw]   += 1.0

    contador = np.where(contador == 0, 1.0, contador)
    return acumulador / contador


# ===========================================================================
# 3. ENSEMBLE DE 5 MODELOS
# ===========================================================================

def cargar_modelos(dir_modelos: str) -> list:
    rutas = sorted(glob.glob(os.path.join(dir_modelos, "fold_*.keras")))
    if len(rutas) == 0:
        raise FileNotFoundError(
            f"No se encontraron modelos en '{dir_modelos}'.\n"
            "Asegurate de haber ejecutado train.py primero."
        )
    modelos = []
    for ruta in rutas:
        print(f"  Cargando: {ruta}")
        m = keras.models.load_model(
            ruta, custom_objects={"dice_coef": dice_coef, "dice_loss": dice_loss}
        )
        modelos.append(m)
    print(f"[INFO] {len(modelos)} modelo(s) cargado(s) correctamente.")
    return modelos


def predecir_ensemble(modelos, imagen, patch_size=(128, 128), stride=64):
    suma = np.zeros(imagen.shape[:2], dtype=np.float32)
    for modelo in modelos:
        suma += predecir_imagen_por_parches(modelo, imagen, patch_size, stride)
    return suma / len(modelos)


# ===========================================================================
# 4. CARGA DE RUTAS DEL CONJUNTO DE TEST
# Empareja por número inicial: "01_test.tif" <-> "01_manual1.gif" -> clave "01"
# ===========================================================================

def cargar_rutas_test(dir_imgs: str, dir_masks: str, dir_fov: str):
    rutas_imgs, rutas_masks, rutas_fov = [], [], []

    for ext in EXTENSIONES:
        rutas_imgs  += glob.glob(os.path.join(dir_imgs,  ext))
        rutas_masks += glob.glob(os.path.join(dir_masks, ext))
        rutas_fov   += glob.glob(os.path.join(dir_fov,   ext))

    # Indexar por número inicial del fichero
    idx_imgs  = {extraer_numero(os.path.splitext(os.path.basename(r))[0]): r for r in rutas_imgs}
    idx_masks = {extraer_numero(os.path.splitext(os.path.basename(r))[0]): r for r in rutas_masks}
    idx_fov   = {extraer_numero(os.path.splitext(os.path.basename(r))[0]): r for r in rutas_fov}

    numeros_comunes = sorted(set(idx_imgs) & set(idx_masks))
    if len(numeros_comunes) == 0:
        raise FileNotFoundError(
            "No se encontraron pares imagen/mascara en las carpetas de test.\n"
            f"  Imagenes en  : {os.path.abspath(dir_imgs)}\n"
            f"  Mascaras en  : {os.path.abspath(dir_masks)}\n"
            f"  Claves imgs  : {sorted(idx_imgs.keys())}\n"
            f"  Claves masks : {sorted(idx_masks.keys())}"
        )

    imgs_ord  = [idx_imgs[n]          for n in numeros_comunes]
    masks_ord = [idx_masks[n]         for n in numeros_comunes]
    fov_ord   = [idx_fov.get(n, None) for n in numeros_comunes]

    print(f"[INFO] {len(imgs_ord)} pares de test encontrados.")
    for n, img, mask in zip(numeros_comunes, imgs_ord, masks_ord):
        print(f"  [{n}]  {os.path.basename(img)}  <->  {os.path.basename(mask)}")

    return imgs_ord, masks_ord, fov_ord


# ===========================================================================
# 5. EVALUACIÓN PRINCIPAL
# ===========================================================================

def evaluar(dir_imgs, dir_masks, dir_fov, dir_modelos, dir_salida,
            patch_size=(128, 128), stride=64, umbral=UMBRAL):

    modelos = cargar_modelos(dir_modelos)
    rutas_imgs, rutas_masks, rutas_fov = cargar_rutas_test(dir_imgs, dir_masks, dir_fov)

    scores_dice = []

    for i, (ruta_img, ruta_mask, ruta_fov) in enumerate(
            zip(rutas_imgs, rutas_masks, rutas_fov)):

        nombre = os.path.splitext(os.path.basename(ruta_img))[0]
        print(f"\n[{i+1}/{len(rutas_imgs)}] Procesando: {nombre}")

        imagen   = leer_imagen_rgb(ruta_img)
        gt_mask  = leer_mascara(ruta_mask)
        fov_mask = leer_fov(ruta_fov) if ruta_fov else None

        mapa_prob    = predecir_ensemble(modelos, imagen, patch_size, stride)
        pred_binaria = (mapa_prob > umbral).astype(np.float32)

        if fov_mask is not None:
            if fov_mask.shape != pred_binaria.shape:
                fov_mask = cv2.resize(
                    fov_mask, (pred_binaria.shape[1], pred_binaria.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            pred_eval = pred_binaria * fov_mask
            gt_eval   = gt_mask      * fov_mask
        else:
            pred_eval = pred_binaria
            gt_eval   = gt_mask

        interseccion = np.sum(pred_eval * gt_eval)
        denominador  = np.sum(pred_eval) + np.sum(gt_eval)
        dice = 1.0 if denominador == 0 else (2.0 * interseccion + 1e-6) / (denominador + 1e-6)

        scores_dice.append(dice)
        print(f"  DICE Score: {dice:.4f}")

        cv2.imwrite(
            os.path.join(dir_salida, f"{nombre}_pred.png"),
            (pred_binaria * 255).astype(np.uint8)
        )

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"{nombre}  -  DICE: {dice:.4f}", fontsize=13)
        axes[0].imshow(imagen);               axes[0].set_title("Imagen original"); axes[0].axis("off")
        axes[1].imshow(gt_mask, cmap="gray"); axes[1].set_title("Ground Truth");    axes[1].axis("off")
        axes[2].imshow(pred_binaria, cmap="gray")
        axes[2].set_title(f"Prediccion (umbral={umbral})");                         axes[2].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(dir_salida, f"{nombre}_comparacion.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ===========================================================================
    # 6. RESUMEN FINAL
    # ===========================================================================
    nombres_base = [os.path.splitext(os.path.basename(r))[0] for r in rutas_imgs]
    media_dice   = np.mean(scores_dice)
    std_dice     = np.std(scores_dice)
    aprobados    = sum(1 for d in scores_dice if d >= 0.75)

    print("\n" + "=" * 60)
    print("  RESULTADOS FINALES - CONJUNTO DE TEST")
    print("=" * 60)
    for nombre_base, dice in zip(nombres_base, scores_dice):
        estado = "OK" if dice >= 0.75 else "!!"
        print(f"  [{estado}]  {nombre_base}: DICE = {dice:.4f}")
    print("=" * 60)
    print(f"  Media DICE       : {media_dice:.4f}")
    print(f"  Desv. tipica     : {std_dice:.4f}")
    print(f"  Imagenes >= 0.75 : {aprobados}/{len(scores_dice)}")
    if media_dice >= 0.75:
        print("  >>> UMBRAL DE 0.75 SUPERADO - PROYECTO APROBADO <<<")
    else:
        print(f"  >>> Umbral no alcanzado (faltan {0.75 - media_dice:.4f} puntos) <<<")
    print("=" * 60)
    print(f"\n[INFO] Resultados guardados en '{dir_salida}/'")

    return scores_dice, media_dice


# ===========================================================================
# 7. PUNTO DE ENTRADA
# ===========================================================================

if __name__ == "__main__":
    print("[INFO] Iniciando evaluacion sobre el conjunto de TEST...")
    evaluar(
        dir_imgs    = DIR_IMAGENES_TEST,
        dir_masks   = DIR_MASCARAS_TEST,
        dir_fov     = DIR_FOV_TEST,
        dir_modelos = DIR_MODELOS,
        dir_salida  = DIR_PREDICCIONES,
        patch_size  = PATCH_SIZE,
        stride      = 64,
        umbral      = UMBRAL
    )