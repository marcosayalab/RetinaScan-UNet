import numpy as np
import cv2
import random
from tensorflow import keras
from PIL import Image

class DataGenerator(keras.utils.PyDataset):
    def __init__(self, rutas_imagenes, rutas_mascaras, batch_size=4, patch_size=(128, 128), workers=1, use_multiprocessing=False):
        """
        Constructor del generador.
        Llamamos al constructor padre de PyDataset para habilitar el multiprocesamiento.
        """
        super().__init__(workers=workers, use_multiprocessing=use_multiprocessing)
        self.rutas_imagenes = rutas_imagenes
        self.rutas_mascaras = rutas_mascaras
        self.batch_size = batch_size
        self.patch_size = patch_size

    def __len__(self):
        """
        Calcula cuántos lotes (batches) componen una época entera.
        """
        return int(np.floor(len(self.rutas_imagenes) / self.batch_size))

    def __getitem__(self, index):
        """
        Se ejecuta cada vez que la red (U-Net) pide un nuevo lote de datos para entrenar.
        """
        # 1. Extraer las rutas específicas para este lote
        rutas_batch_x = self.rutas_imagenes[index * self.batch_size : (index + 1) * self.batch_size]
        rutas_batch_y = self.rutas_mascaras[index * self.batch_size : (index + 1) * self.batch_size]

        # 2. Inicializar matrices vacías para almacenar los parches recortados
        # X tendrá forma (batch_size, 128, 128, 3) -> 3 canales de color (RGB)
        # y tendrá forma (batch_size, 128, 128, 1) -> 1 canal (Blanco y negro)
        X = np.empty((self.batch_size, *self.patch_size, 3), dtype=np.float32)
        y = np.empty((self.batch_size, *self.patch_size, 1), dtype=np.float32)

        # 3. Procesar cada imagen del lote actual
        for i, (ruta_img, ruta_mask) in enumerate(zip(rutas_batch_x, rutas_batch_y)):
            
            # 1: Leer las imágenes del disco duro
            # Pista: Usa cv2.imread(). ¡Cuidado! OpenCV lee en BGR, quizás quieras pasarlo a RGB.
            # Para la máscara, léela en escala de grises (cv2.IMREAD_GRAYSCALE).
            ojo_bgr = cv2.imread(ruta_img) 
            ojo_rgb = cv2.cvtColor(ojo_bgr, cv2.COLOR_BGR2RGB)
            mask_pil = Image.open(ruta_mask)
            mask_array = np.array(mask_pil)

            # 2: Normalización
            # Las imágenes vienen con píxeles de 0 a 255. Divídelas para que estén entre 0.0 y 1.0.
            img_normalizada = ojo_rgb / 255.0
            mask_normalizada = mask_array / 255.0

            # 3: El Parcheado (Troceado coordinado)
            # Tienes que generar unas coordenadas X e Y aleatorias.
            # Asegúrate de que las coordenadas no se salgan de los bordes de la imagen original
            # al sumarle los 128 píxeles del patch_size.
            # Luego, recorta EXACTAMENTE las mismas coordenadas tanto en 'img' como en 'mask'.

            alto_original, ancho_original = img_normalizada.shape[:2]
            limite_y = alto_original - self.patch_size[0]
            limite_x = ancho_original - self.patch_size[1]

            x_random = random.randint(0, limite_x)
            y_random = random.randint(0, limite_y)

            parche_img = img_normalizada[y_random : y_random + self.patch_size[0], x_random : x_random + self.patch_size[1]]
            parche_mask = mask_normalizada[y_random : y_random + self.patch_size[0], x_random : x_random + self.patch_size[1]]

            # 4: Data Augmentation (Opcional pero muy recomendado)
            # Genera un número aleatorio. Si es mayor que 0.5, dale la vuelta horizontalmente (cv2.flip)
            # tanto al parche de la imagen como al de la máscara.
            if random.random() > 0.5: 
                parche_img = cv2.flip(parche_img, 1)  # 1 para flip horizontal
                parche_mask = cv2.flip(parche_mask, 1)

            # 5. Guardar los parches finales en las matrices del batch
            X[i,] = parche_img
            
            # OpenCV devuelve matrices de (128, 128) al leer en escala de grises. 
            # Keras necesita (128, 128, 1). Usamos np.expand_dims para añadir esa dimensión extra.
            y[i,] = np.expand_dims(parche_mask, axis=-1)

        return X, y