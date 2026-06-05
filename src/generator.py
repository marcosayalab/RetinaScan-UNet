import numpy as np
import cv2
import random
import keras

class DataGenerator(keras.utils.PyDataset):
    def __init__(self, rutas_imagenes, rutas_gt, batch_size=4, patch_size=(128, 128), workers=1, use_multiprocessing=False):
        """
        Constructor del generador.
        
        NOTA SOBRE TERMINOLOGÍA:
        - 'gt' = Ground Truth (verdad fundamental)
        - Representa las segmentaciones manuales de referencia anotadas por expertos
        - En el dataset DRIVE se encuentran en las carpetas '1st_manual' y '2nd_manual'
        - Las carpetas 'mask' contienen solo máscaras del campo de visión (FoV), no segmentaciones
        
        Llamamos al constructor padre de PyDataset para habilitar el multiprocesamiento.
        """
        super().__init__(workers=workers, use_multiprocessing=use_multiprocessing)
        self.rutas_imagenes = rutas_imagenes
        self.rutas_gt = rutas_gt
        self.batch_size = batch_size
        self.patch_size = patch_size

    def __len__(self):
        return int(np.floor(len(self.rutas_imagenes) / self.batch_size))

    def __getitem__(self, index):
        
        # 1. Extraer las rutas específicas para este lote
        rutas_batch_x = self.rutas_imagenes[index * self.batch_size : (index + 1) * self.batch_size]
        rutas_batch_y = self.rutas_gt[index * self.batch_size : (index + 1) * self.batch_size]

        # 2. Inicializar matrices vacías para almacenar los parches recortados
        # X tendrá forma (batch_size, 128, 128, 3) -> 3 canales de color (RGB)
        # y tendrá forma (batch_size, 128, 128, 1) -> 1 canal (Blanco y negro con ground truth)
        X = np.empty((self.batch_size, *self.patch_size, 3), dtype=np.float32)
        y = np.empty((self.batch_size, *self.patch_size, 1), dtype=np.float32)

        # 3. Procesar cada imagen del lote actual
        for i, (ruta_img, ruta_gt) in enumerate(zip(rutas_batch_x, rutas_batch_y)):
            
            ojo_bgr = cv2.imread(ruta_img)
            if ojo_bgr is None:
                raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta_img}. Verifica que la ruta existe.")
            
            ojo_rgb = cv2.cvtColor(ojo_bgr, cv2.COLOR_BGR2RGB)

            # FILTRO CLAHE
            # Pasamos a espacio de color LAB (Luminosidad, A, B)
            ojo_lab = cv2.cvtColor(ojo_rgb, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # Se lo aplicamos solo al canal de luminosidad (L) para no distorsionar colores
            ojo_lab[:, :, 0] = clahe.apply(ojo_lab[:, :, 0])
            ojo_rgb = cv2.cvtColor(ojo_lab, cv2.COLOR_LAB2RGB)

            # Para el ground truth, se lee en escala de grises (cv2.IMREAD_GRAYSCALE).
            gt_array = cv2.imread(ruta_gt, cv2.IMREAD_GRAYSCALE)
            if gt_array is None:
                raise FileNotFoundError(f"No se pudo cargar el ground truth: {ruta_gt}. Verifica que la ruta existe.")

            
            img_normalizada = ojo_rgb / 255.0
            gt_normalizada = gt_array / 255.0

            # Parcheado 
            # Generar unas coordenadas X e Y aleatorias.
            # Asegurarse de que las coordenadas no se salgan de los bordes de la imagen original al sumarle los 128 píxeles del patch_size.
            # Recortar EXACTAMENTE las mismas coordenadas tanto en 'img' como en 'gt'.

            alto_original, ancho_original = img_normalizada.shape[:2]
            limite_y = alto_original - self.patch_size[0]
            limite_x = ancho_original - self.patch_size[1]

            x_random = random.randint(0, limite_x)
            y_random = random.randint(0, limite_y)

            parche_img = img_normalizada[y_random : y_random + self.patch_size[0], x_random : x_random + self.patch_size[1]]
            parche_gt = gt_normalizada[y_random : y_random + self.patch_size[0], x_random : x_random + self.patch_size[1]]

            # Data Augmentation 
            # Volteo horizontal
            
            if random.random() > 0.5: 
                parche_img = cv2.flip(parche_img, 1)  # 1 para flip horizontal
                parche_gt = cv2.flip(parche_gt, 1)

            # Volteo vertical
            if random.random() > 0.5:
                parche_img = cv2.flip(parche_img, 0)
                parche_gt = cv2.flip(parche_gt, 0)

            # Rotaciones aleatorias de 90, 180 o 270 grados
            if random.random() > 0.5:
                k = random.choice([0, 1, 2])  # 0=90°, 1=180°, 2=270°
                parche_img = cv2.rotate(parche_img, k)
                parche_gt = cv2.rotate(parche_gt, k)
            
            # Alterción aleatoria del brillo (solo en la imagen, no en el ground truth)
            if random.random() > 0.5:
                factor_brillo = random.uniform(0.8, 1.2)  # Brillo entre 80% y 120%
                parche_img = np.clip(parche_img * factor_brillo, 0.0, 1.0)  # Asegura que los valores sigan entre 0 y 1

            # Ruido Gausiano (simula mala calidad de cámara)
            if random.random() > 0.5:
                ruido = np.random.normal(0, 0.05, parche_img.shape)  # Media=0, Desviación=0.05
                parche_img = np.clip(parche_img + ruido, 0.0, 1.0)
            
            # Guardar los parches finales en las matrices del batch
            X[i,] = parche_img
            
            y[i,] = np.expand_dims(parche_gt, axis=-1)

        return X, y