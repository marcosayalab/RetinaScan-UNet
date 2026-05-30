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
import keras

from sklearn.metrics import precision_score, recall_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from metrics import dice_coef, dice_loss, bce_dice_loss


# ===========================================================================
# 0. CONFIGURACIÓN CENTRAL
# ===========================================================================

DIR_IMAGENES_TEST  = os.path.join("data", "test", "images")
DIR_GT1_TEST        = os.path.join("data", "test", "1st_manual")
DIR_GT2_TEST        = os.path.join("data", "test", "2nd_manual")
# NOTA: 'mask' aquí específicamente se refiere a la máscara del campo de visión (FoV mask)
# Indica qué píxeles están dentro del campo de visión del retinógrafo
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


def leer_gt(ruta: str) -> np.ndarray:
    gt = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(f"No se pudo leer el ground truth: {ruta}")
    return (gt > 127).astype(np.float32)


def leer_fov(ruta: str) -> np.ndarray:
    fov = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if fov is None:
        return None
    return (fov > 127).astype(np.float32)


def extraer_numero(nombre_base: str) -> str:
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
# Ahora carga ambos ground truths (1st_manual y 2nd_manual)
# ===========================================================================

def cargar_rutas_test(dir_imgs: str, dir_gt1: str, dir_gt2: str, dir_fov: str):
    rutas_imgs, rutas_gt1, rutas_gt2, rutas_fov = [], [], [], []

    for ext in EXTENSIONES:
        rutas_imgs += glob.glob(os.path.join(dir_imgs,   ext))
        rutas_gt1  += glob.glob(os.path.join(dir_gt1,  ext))
        rutas_gt2  += glob.glob(os.path.join(dir_gt2,  ext))
        rutas_fov  += glob.glob(os.path.join(dir_fov,   ext))

    # Indexar por número inicial del fichero
    idx_imgs = {extraer_numero(os.path.splitext(os.path.basename(r))[0]): r for r in rutas_imgs}
    idx_gt1  = {extraer_numero(os.path.splitext(os.path.basename(r))[0]): r for r in rutas_gt1}
    idx_gt2  = {extraer_numero(os.path.splitext(os.path.basename(r))[0]): r for r in rutas_gt2}
    idx_fov  = {extraer_numero(os.path.splitext(os.path.basename(r))[0]): r for r in rutas_fov}

    numeros_comunes = sorted(set(idx_imgs) & set(idx_gt1) & set(idx_gt2))
    if len(numeros_comunes) == 0:
        raise FileNotFoundError(
            "No se encontraron pares imagen/ground_truth en las carpetas de test.\n"
            f"  Imagenes en      : {os.path.abspath(dir_imgs)}\n"
            f"  Ground truth 1   : {os.path.abspath(dir_gt1)}\n"
            f"  Ground truth 2   : {os.path.abspath(dir_gt2)}\n"
            f"  Claves imgs      : {sorted(idx_imgs.keys())}\n"
            f"  Claves gt1       : {sorted(idx_gt1.keys())}\n"
            f"  Claves gt2       : {sorted(idx_gt2.keys())}"
        )

    imgs_ord = [idx_imgs[n]          for n in numeros_comunes]
    gt1_ord  = [idx_gt1[n]           for n in numeros_comunes]
    gt2_ord  = [idx_gt2[n]           for n in numeros_comunes]
    fov_ord  = [idx_fov.get(n, None) for n in numeros_comunes]

    print(f"[INFO] {len(imgs_ord)} pares de test encontrados.")
    for n, img, gt1, gt2 in zip(numeros_comunes, imgs_ord, gt1_ord, gt2_ord):
        print(f"  [{n}]  {os.path.basename(img)}  <->  {os.path.basename(gt1)}  |  {os.path.basename(gt2)}")

    return imgs_ord, gt1_ord, gt2_ord, fov_ord


# ===========================================================================
# 5. EVALUACIÓN PRINCIPAL
# ===========================================================================

def evaluar(dir_imgs, dir_gt1, dir_gt2, dir_fov, dir_modelos, dir_salida,
            patch_size=(128, 128), stride=64, umbral=UMBRAL):

    modelos = cargar_modelos(dir_modelos)
    rutas_imgs, rutas_gt1, rutas_gt2, rutas_fov = cargar_rutas_test(dir_imgs, dir_gt1, dir_gt2, dir_fov)

    scores_dice_promedio = []  # DICE promedio de ambos expertos
    scores_dice_experto1 = []  # DICE del experto 1
    scores_dice_experto2 = []  # DICE del experto 2
    scores_precision = []
    scores_recall = []

    for i, (ruta_img, ruta_gt1, ruta_gt2, ruta_fov) in enumerate(
            zip(rutas_imgs, rutas_gt1, rutas_gt2, rutas_fov)):

        nombre = os.path.splitext(os.path.basename(ruta_img))[0]
        print(f"\n[{i+1}/{len(rutas_imgs)}] Procesando: {nombre}")

        imagen    = leer_imagen_rgb(ruta_img)
        gt1_mask  = leer_gt(ruta_gt1)
        gt2_mask  = leer_gt(ruta_gt2)
        fov_mask  = leer_fov(ruta_fov) if ruta_fov else None

        mapa_prob    = predecir_ensemble(modelos, imagen, patch_size, stride)
        pred_binaria = (mapa_prob > umbral).astype(np.float32)

        if fov_mask is not None:
            if fov_mask.shape != pred_binaria.shape:
                fov_mask = cv2.resize(
                    fov_mask, (pred_binaria.shape[1], pred_binaria.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            pred_eval  = pred_binaria * fov_mask
            gt1_eval   = gt1_mask    * fov_mask
            gt2_eval   = gt2_mask    * fov_mask
        else:
            pred_eval  = pred_binaria
            gt1_eval   = gt1_mask
            gt2_eval   = gt2_mask

        # CALCULAR DICE PARA AMBOS EXPERTOS
        interseccion1 = np.sum(pred_eval * gt1_eval)
        denominador1  = np.sum(pred_eval) + np.sum(gt1_eval)
        dice1 = 1.0 if denominador1 == 0 else (2.0 * interseccion1 + 1e-6) / (denominador1 + 1e-6)
        
        interseccion2 = np.sum(pred_eval * gt2_eval)
        denominador2  = np.sum(pred_eval) + np.sum(gt2_eval)
        dice2 = 1.0 if denominador2 == 0 else (2.0 * interseccion2 + 1e-6) / (denominador2 + 1e-6)
        
        # PROMEDIO DE DICE DE AMBOS EXPERTOS
        dice_promedio = (dice1 + dice2) / 2.0
        
        scores_dice_experto1.append(dice1)
        scores_dice_experto2.append(dice2)
        scores_dice_promedio.append(dice_promedio)

        # PRECISIÓN Y RECALL (usando promedio de ambos GTs)
        # 1. Aplastamos las imágenes a vectores 1D
        gt_promedio = (gt1_eval + gt2_eval) / 2.0
        gt_plano = (gt_promedio > 0.5).astype(np.float32).flatten()
        pred_plano = pred_eval.flatten()
            
        # 2. Calculamos las métricas (zero_division=0 evita errores si la IA predice todo negro)
        precision = precision_score(gt_plano, pred_plano, zero_division=0)
        recall = recall_score(gt_plano, pred_plano, zero_division=0)
            
        # 3. Guardamos los resultados 
        scores_precision.append(precision)
        scores_recall.append(recall)

        print(f"  DICE Exp1: {dice1:.4f} | DICE Exp2: {dice2:.4f} | DICE Promedio: {dice_promedio:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        cv2.imwrite(
            os.path.join(dir_salida, f"{nombre}_pred.png"),
            (pred_binaria * 255).astype(np.uint8)
        )

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(f"{nombre}  -  DICE Exp1: {dice1:.4f} | Exp2: {dice2:.4f} | Promedio: {dice_promedio:.4f}", fontsize=12)
        axes[0, 0].imshow(imagen);                        axes[0, 0].set_title("Imagen original");     axes[0, 0].axis("off")
        axes[0, 1].imshow(gt1_mask, cmap="gray");        axes[0, 1].set_title("GT Experto 1");       axes[0, 1].axis("off")
        axes[1, 0].imshow(gt2_mask, cmap="gray");        axes[1, 0].set_title("GT Experto 2");       axes[1, 0].axis("off")
        axes[1, 1].imshow(pred_eval, cmap="gray");       axes[1, 1].set_title(f"Prediccion (umbral={umbral})"); axes[1, 1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(dir_salida, f"{nombre}_comparacion.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    # ===========================================================================
    # 6. RESUMEN FINAL
    # ===========================================================================
    nombres_base = [os.path.splitext(os.path.basename(r))[0] for r in rutas_imgs]
    media_dice_exp1 = np.mean(scores_dice_experto1)
    media_dice_exp2 = np.mean(scores_dice_experto2)
    media_dice_promedio = np.mean(scores_dice_promedio)
    media_precision = np.mean(scores_precision)
    media_recall = np.mean(scores_recall)
    std_dice = np.std(scores_dice_promedio)
    aprobados = sum(1 for d in scores_dice_promedio if d >= 0.75)

    print("\n" + "=" * 80)
    print("  RESULTADOS FINALES - CONJUNTO DE TEST (CON AMBOS EXPERTOS)")
    print("=" * 80)
    for nombre_base, dice1, dice2, dice_prom in zip(nombres_base, scores_dice_experto1, scores_dice_experto2, scores_dice_promedio):
        estado = "OK" if dice_prom >= 0.75 else "!!"
        print(f"  [{estado}]  {nombre_base}: DICE_Exp1={dice1:.4f} | DICE_Exp2={dice2:.4f} | Promedio={dice_prom:.4f}")
    print("=" * 80)
    print(f"  Media DICE (Experto 1) : {media_dice_exp1:.4f}")
    print(f"  Media DICE (Experto 2) : {media_dice_exp2:.4f}")
    print(f"  Media DICE (Promedio)  : {media_dice_promedio:.4f}")
    print(f"  Media Precisión        : {media_precision:.4f}")
    print(f"  Media Recall           : {media_recall:.4f}")
    print(f"  Desv. típica           : {std_dice:.4f}")
    print(f"  Imágenes >= 0.75       : {aprobados}/{len(scores_dice_promedio)}")
    if media_dice_promedio >= 0.75:
        print("\n  >>> UMBRAL DE 0.75 SUPERADO - PROYECTO APROBADO <<<")
    else:
        print(f"\n  >>> Umbral no alcanzado (faltan {0.75 - media_dice_promedio:.4f} puntos) <<<")
    print("=" * 80)
    print(f"\n[INFO] Resultados guardados en '{dir_salida}/'")

    return scores_dice_promedio, media_dice_promedio


# ===========================================================================
# 7. PUNTO DE ENTRADA
# ===========================================================================

if __name__ == "__main__":
    print("[INFO] Iniciando evaluacion sobre el conjunto de TEST...")
    evaluar(
        dir_imgs    = DIR_IMAGENES_TEST,
        dir_gt1     = DIR_GT1_TEST,
        dir_gt2     = DIR_GT2_TEST,
        dir_fov     = DIR_FOV_TEST,
        dir_modelos = DIR_MODELOS,
        dir_salida  = DIR_PREDICCIONES,
        patch_size  = PATCH_SIZE,
        stride      = 64,
        umbral      = UMBRAL
    )