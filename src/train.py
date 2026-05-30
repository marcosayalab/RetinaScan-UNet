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

img_dir = "data/training/images"
gt_dir = "data/training/1st_manual"
models_dir = "models"

os.makedirs(models_dir, exist_ok=True)

batch_size = 4
patch_size = (128, 128)
epochs = 200
n_folds = 5
learning_rate = 0.0001
patience = 25
input_shape = (128, 128, 3)
reps = 10
seed = 42

np.random.seed(seed)
tf.random.set_seed(seed)

def get_paths(dir_img, dir_gt):
    exts = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"]
    imgs = []
    gts = []

    for ext in exts:
        imgs += glob.glob(f"{dir_img}/{ext}")
        gts += glob.glob(f"{dir_gt}/{ext}")

    imgs = sorted(set(imgs))
    gts = sorted(set(gts))

    if not imgs or not gts or len(imgs) != len(gts):
        raise ValueError("Error cargando los archivos. Revisa las carpetas y que haya el mismo número de imágenes y máscaras.")

    print(f"Cargados {len(imgs)} pares de imágenes.")
    return np.array(imgs), np.array(gts)

def get_callbacks(fold):
    ruta_modelo = f"{models_dir}/fold_{fold}.keras"
    
    return [
        EarlyStopping(monitor="val_dice_coef", mode="max", patience=patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=ruta_modelo, monitor="val_dice_coef", mode="max", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-7, verbose=1)
    ]

def train(imgs, gts):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    resultados = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(imgs), 1):
        print(f"\n--- Empezando fold {fold}/{n_folds} ---")
        
        gen_train = DataGenerator(
            rutas_imagenes=list(imgs[train_idx]) * reps,
            rutas_gt=list(gts[train_idx]) * reps,
            batch_size=batch_size,
            patch_size=patch_size
        )
        
        gen_val = DataGenerator(
            rutas_imagenes=list(imgs[val_idx]),
            rutas_gt=list(gts[val_idx]),
            batch_size=batch_size,
            patch_size=patch_size
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
            callbacks=get_callbacks(fold),
            verbose=1
        )

        mejor_val = max(historial.history["val_dice_coef"])
        resultados.append(mejor_val)
        print(f"Fold {fold} terminado con DICE de {mejor_val:.4f}")

    print("\nResumen final:")
    for i, res in enumerate(resultados, 1):
        print(f"Fold {i}: {res:.4f}")
    print(f"Media: {np.mean(resultados):.4f} | Desviación: {np.std(resultados):.4f}")


if __name__ == "__main__":
    print("Arrancando script...")
    imgs, gts = get_paths(img_dir, gt_dir)

    print("\nComprobando que emparejan bien:")
    for i, g in zip(imgs[:3], gts[:3]):
        print(f" - {os.path.basename(i)} -> {os.path.basename(g)}")

    train(imgs, gts)