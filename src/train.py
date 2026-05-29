import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold

from generator import DataGenerator
from model import construir_unet
from metrics import dice_coef, dice_loss, bce_dice_loss

# ===========================================================================
# 0. CONFIGURACIÓN
# ===========================================================================
DIR_IMAGENES_TRAIN = os.path.join("data", "training", "images")
# NOTA: 'gt' = Ground Truth (segmentaciones manuales de referencia)
DIR_GT_TRAIN       = os.path.join("data", "training", "1st_manual")
DIR_MODELOS        = "modelos_guardados"
os.makedirs(DIR_MODELOS, exist_ok=True)

BATCH_SIZE    = 4
PATCH_SIZE    = (128, 128)
EPOCHS        = 200
N_FOLDS       = 5
LEARNING_RATE = 1e-4
PATIENCE_EARLY = 25
INPUT_SHAPE   = (128, 128, 3)
REPETICIONES  = 10   # multiplica parches por época
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ===========================================================================
# 1. CARGA DE RUTAS
# ===========================================================================
def cargar_rutas_drive(dir_imagenes, dir_gt):
    """Carga rutas de imágenes y ground truth desde el dataset DRIVE"""
    EXTENSIONES = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")
    rutas_imgs, rutas_gt = [], []
    for ext in EXTENSIONES:
        rutas_imgs += glob.glob(os.path.join(dir_imagenes, ext))
        rutas_gt  += glob.glob(os.path.join(dir_gt, ext))
    rutas_imgs = sorted(set(rutas_imgs))
    rutas_gt   = sorted(set(rutas_gt))

    if len(rutas_imgs) == 0:
        raise FileNotFoundError(f"No se encontraron imágenes en '{dir_imagenes}'.")
    if len(rutas_gt) == 0:
        raise FileNotFoundError(f"No se encontraron ground truth en '{dir_gt}'.")
    if len(rutas_imgs) != len(rutas_gt):
        raise ValueError(f"Nº imágenes ({len(rutas_imgs)}) ≠ Nº ground truth ({len(rutas_gt)}).")

    print(f"[INFO] {len(rutas_imgs)} pares imagen/ground_truth cargados.")
    return np.array(rutas_imgs), np.array(rutas_gt)

# ===========================================================================
# 2. CALLBACKS
# ===========================================================================
def crear_callbacks(fold):
    ruta_modelo = os.path.join(DIR_MODELOS, f"fold_{fold + 1}.keras")
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_dice_coef", mode="max",
            patience=PATIENCE_EARLY, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=ruta_modelo, monitor="val_dice_coef",
            mode="max", save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-7, verbose=1
        ),
    ]

# ===========================================================================
# 3. ENTRENAMIENTO K-FOLD
# ===========================================================================
def entrenar_con_kfold(rutas_imgs, rutas_gt):
    """Entrena la red con validación cruzada k-fold"""
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    historial_dice_val = []

    for fold, (indices_train, indices_val) in enumerate(kf.split(rutas_imgs)):
        print("\n" + "=" * 60)
        print(f"  PLIEGUE {fold + 1} / {N_FOLDS}")
        print(f"  Train: {len(indices_train)} imgs | Val: {len(indices_val)} imgs")
        print("=" * 60)

        gen_train = DataGenerator(
            rutas_imagenes=list(rutas_imgs[indices_train]) * REPETICIONES,
            rutas_gt=list(rutas_gt[indices_train]) * REPETICIONES,
            batch_size=BATCH_SIZE,
            patch_size=PATCH_SIZE
        )
        gen_val = DataGenerator(
            rutas_imagenes=list(rutas_imgs[indices_val]),
            rutas_gt=list(rutas_gt[indices_val]),
            batch_size=BATCH_SIZE,
            patch_size=PATCH_SIZE
        )

        modelo = construir_unet(input_shape=INPUT_SHAPE)
        modelo.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=bce_dice_loss,
            metrics=[dice_coef]
        )

        historia = modelo.fit(
            gen_train,
            validation_data=gen_val,
            epochs=EPOCHS,
            callbacks=crear_callbacks(fold),
            verbose=1
        )

        mejor_dice = max(historia.history["val_dice_coef"])
        historial_dice_val.append(mejor_dice)
        print(f"\n[FOLD {fold + 1}] Mejor DICE validación: {mejor_dice:.4f}")

    print("\n" + "=" * 60)
    print("  RESUMEN FINAL - VALIDACIÓN CRUZADA 5-FOLD")
    print("=" * 60)
    for i, dice in enumerate(historial_dice_val):
        print(f"  Fold {i + 1}: DICE = {dice:.4f}")
    print(f"\n  Media DICE : {np.mean(historial_dice_val):.4f}")
    print(f"  Desv. típ. : {np.std(historial_dice_val):.4f}")
    print("=" * 60)

# ===========================================================================
# 4. PUNTO DE ENTRADA
# ===========================================================================
if __name__ == "__main__":
    print("[INFO] Cargando rutas del dataset DRIVE...")
    rutas_imgs, rutas_gt = cargar_rutas_drive(DIR_IMAGENES_TRAIN, DIR_GT_TRAIN)

    print("\n--- DIAGNÓSTICO DE RUTAS ---")
    for img, gt in zip(rutas_imgs[:3], rutas_gt[:3]):
        print(f"  IMG : {os.path.basename(img)}")
        print(f"  GT : {os.path.basename(gt)}")

    print("\n[INFO] Iniciando entrenamiento con validación cruzada 5-Fold...")
    entrenar_con_kfold(rutas_imgs, rutas_gt)