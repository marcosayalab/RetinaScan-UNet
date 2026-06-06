import numpy as np
import cv2
import random
import keras


class DataGenerator(keras.utils.PyDataset):
    def __init__(self, rutas_imagenes, rutas_gt, rutas_fov=None,
                 batch_size=4, patch_size=(128, 128),
                 workers=1, use_multiprocessing=False):
        super().__init__(workers=workers, use_multiprocessing=use_multiprocessing)
        self.rutas_imagenes = rutas_imagenes
        self.rutas_gt = rutas_gt
        self.rutas_fov = rutas_fov  # None = sin máscara FoV
        self.batch_size = batch_size
        self.patch_size = patch_size

    def __len__(self):
        return int(np.floor(len(self.rutas_imagenes) / self.batch_size))

    def __getitem__(self, index):
        rutas_batch_x = self.rutas_imagenes[index * self.batch_size : (index + 1) * self.batch_size]
        rutas_batch_y = self.rutas_gt[index * self.batch_size : (index + 1) * self.batch_size]

        # Extraer rutas FoV del batch si están disponibles
        if self.rutas_fov is not None:
            rutas_batch_fov = self.rutas_fov[index * self.batch_size : (index + 1) * self.batch_size]
        else:
            rutas_batch_fov = [None] * self.batch_size

        X = np.empty((self.batch_size, *self.patch_size, 1), dtype=np.float32)
        y = np.empty((self.batch_size, *self.patch_size, 1), dtype=np.float32)

        for i, (ruta_img, ruta_gt, ruta_fov) in enumerate(zip(rutas_batch_x, rutas_batch_y, rutas_batch_fov)):

            ojo_bgr = cv2.imread(ruta_img)
            if ojo_bgr is None:
                raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta_img}")

            ojo_rgb = cv2.cvtColor(ojo_bgr, cv2.COLOR_BGR2RGB)

            # CLAHE sobre canal L
            ojo_lab = cv2.cvtColor(ojo_rgb, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            ojo_lab[:, :, 0] = clahe.apply(ojo_lab[:, :, 0])
            ojo_rgb = cv2.cvtColor(ojo_lab, cv2.COLOR_LAB2RGB)

            # Convertir a escala de grises
            ojo_gray = cv2.cvtColor(ojo_rgb, cv2.COLOR_RGB2GRAY)

            gt_array = cv2.imread(ruta_gt, cv2.IMREAD_GRAYSCALE)
            if gt_array is None:
                raise FileNotFoundError(f"No se pudo cargar el ground truth: {ruta_gt}")

            img_normalizada = ojo_gray / 255.0
            gt_normalizada  = gt_array / 255.0

            # Aplicar máscara FoV antes del parcheado — elimina el fondo negro
            # exterior del ojo para que el modelo no aprenda sobre esa zona
            if ruta_fov is not None:
                fov = cv2.imread(ruta_fov, cv2.IMREAD_GRAYSCALE)
                if fov is not None:
                    fov_bin = (fov > 127).astype(np.float32)
                    img_normalizada = img_normalizada * fov_bin
                    gt_normalizada  = gt_normalizada  * fov_bin

            # Parcheado aleatorio
            alto_original, ancho_original = img_normalizada.shape[:2]
            limite_y = alto_original - self.patch_size[0]
            limite_x = ancho_original - self.patch_size[1]

            x_random = random.randint(0, limite_x)
            y_random = random.randint(0, limite_y)

            parche_img = img_normalizada[y_random : y_random + self.patch_size[0],
                                         x_random : x_random + self.patch_size[1]]
            parche_gt  = gt_normalizada[y_random : y_random + self.patch_size[0],
                                         x_random : x_random + self.patch_size[1]]

            # --- Data Augmentation ---

            # Volteo horizontal
            if random.random() > 0.5:
                parche_img = cv2.flip(parche_img, 1)
                parche_gt  = cv2.flip(parche_gt,  1)

            # Volteo vertical
            if random.random() > 0.5:
                parche_img = cv2.flip(parche_img, 0)
                parche_gt  = cv2.flip(parche_gt,  0)

            # Rotaciones discretas 90°, 180°, 270°
            if random.random() > 0.5:
                k = random.choice([0, 1, 2])
                parche_img = cv2.rotate(parche_img, k)
                parche_gt  = cv2.rotate(parche_gt,  k)

            # Rotación continua -15° a 15° — simula variación natural de cámara
            if random.random() > 0.5:
                angulo = random.uniform(-15, 15)
                h, w = parche_img.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angulo, 1)
                parche_img = cv2.warpAffine(parche_img, M, (w, h),
                                            flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_REFLECT_101)
                parche_gt  = cv2.warpAffine(parche_gt, M, (w, h),
                                            flags=cv2.INTER_NEAREST,
                                            borderMode=cv2.BORDER_REFLECT_101)

            # Variación de brillo
            if random.random() > 0.5:
                factor_brillo = random.uniform(0.8, 1.2)
                parche_img = np.clip(parche_img * factor_brillo, 0.0, 1.0)

            # Ruido Gaussiano
            if random.random() > 0.5:
                ruido = np.random.normal(0, 0.05, parche_img.shape)
                parche_img = np.clip(parche_img + ruido, 0.0, 1.0)

            X[i,] = np.expand_dims(parche_img, axis=-1)
            y[i,] = np.expand_dims(parche_gt,  axis=-1)

        return X, y