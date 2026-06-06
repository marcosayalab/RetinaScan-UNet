import os
import glob
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import KFold

from generator import DataGenerator
from model import construir_unet
from metrics import dice_coef, bce_dice_loss

img_dir = os.path.join("data", "training", "images")
gt_dir  = os.path.join("data", "training", "1st_manual")
fov_dir = os.path.join("data", "training", "mask")
models_dir = os.path.join("models")

os.makedirs(models_dir, exist_ok=True)

batch_size   = 4
patch_size   = (128, 128)
epochs       = 200
n_folds      = 5
learning_rate = 0.0001
input_shape  = (128, 128, 1)
seed         = 42

FOLD_CONFIG = {
    1: {"reps": 10, "patience": 25},
    2: {"reps": 10, "patience": 25},
    3: {"reps": 20, "patience": 40},
    4: {"reps": 10, "patience": 25},
    5: {"reps": 10, "patience": 25},
}

np.random.seed(seed)
tf.random.set_seed(seed)


def get_paths(dir_img, dir_gt, dir_fov):
    exts = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"]
    imgs, gts, fovs = [], [], []

    for ext in exts:
        imgs += glob.glob(os.path.join(dir_img, ext))
        gts  += glob.glob(os.path.join(dir_gt,  ext))
        fovs += glob.glob(os.path.join(dir_fov, ext))

    imgs = sorted(set(imgs))
    gts  = sorted(set(gts))
    fovs = sorted(set(fovs))

    if not imgs or not gts or len(imgs) != len(gts):
        raise ValueError("Error cargando imágenes. Revisa las carpetas.")
    if len(fovs) != len(imgs):
        raise ValueError(f"Número de máscaras FoV ({len(fovs)}) no coincide con imágenes ({len(imgs)}).")

    print(f"Cargados {len(imgs)} pares de imágenes + máscaras FoV.")
    return np.array(imgs), np.array(gts), np.array(fovs)


def get_callbacks(fold, fold_patience):
    ruta_modelo = os.path.join(models_dir, f"fold_{fold}.keras")
    return [
        EarlyStopping(monitor="val_dice_coef", mode="max",
                      patience=fold_patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=ruta_modelo, monitor="val_dice_coef",
                        mode="max", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=10, min_lr=1e-7, verbose=1),
    ]


def train(imgs, gts, fovs):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    resultados = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(imgs), 1):
        cfg         = FOLD_CONFIG[fold]
        fold_reps   = cfg["reps"]
        fold_patience = cfg["patience"]

        print(f"\n{'='*70}")
        print(f"FOLD {fold}/{n_folds}  |  reps={fold_reps}  |  patience={fold_patience}")
        print(f"{'='*70}")

        gen_train = DataGenerator(
            rutas_imagenes=list(imgs[train_idx]) * fold_reps,
            rutas_gt=list(gts[train_idx]) * fold_reps,
            rutas_fov=list(fovs[train_idx]) * fold_reps,
            batch_size=batch_size,
            patch_size=patch_size,
        )

        gen_val = DataGenerator(
            rutas_imagenes=list(imgs[val_idx]),
            rutas_gt=list(gts[val_idx]),
            rutas_fov=list(fovs[val_idx]),
            batch_size=batch_size,
            patch_size=patch_size,
        )

        modelo = construir_unet(input_shape=input_shape)
        modelo.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=bce_dice_loss,
            metrics=[dice_coef]
        )

        historial = modelo.fit(
            gen_train,
            validation_data=gen_val,
            epochs=epochs,
            callbacks=get_callbacks(fold, fold_patience),
            verbose=1
        )

        mejor_val = max(historial.history["val_dice_coef"])
        resultados.append(mejor_val)
        estado = "OK" if mejor_val >= 0.75 else "FALLO"
        print(f"[{estado}] Fold {fold}: DICE = {mejor_val:.4f}")

    print(f"\n{'='*70}")
    print("RESUMEN FINAL")
    print(f"{'='*70}")
    for i, res in enumerate(resultados, 1):
        estado = "OK" if res >= 0.75 else "FALLO"
        print(f"[{estado}] Fold {i}: {res:.4f}")

    media = np.mean(resultados)
    std   = np.std(resultados)
    aprobados = sum(1 for r in resultados if r >= 0.75)
    print(f"\nMedia: {media:.4f} | Desviación: {std:.4f}")
    print(f"Folds >= 0.75: {aprobados}/{n_folds}")
    print(f"{'='*70}")


if __name__ == "__main__":
    print("Arrancando entrenamiento...")
    imgs, gts, fovs = get_paths(img_dir, gt_dir, fov_dir)

    print("\nComprobando emparejamiento (primeros 3):")
    for i, g, f in zip(imgs[:3], gts[:3], fovs[:3]):
        print(f"  {os.path.basename(i)} -> {os.path.basename(g)} -> {os.path.basename(f)}")

    train(imgs, gts, fovs)