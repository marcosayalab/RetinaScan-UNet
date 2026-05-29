import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

# Importamos el generador (del archivo generator.py)
from generator import DataGenerator
# Importamos EL MODELO (del archivo model.py)
from model import construir_unet

# Construimos las rutas correctas desde la raíz del proyecto
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))

# 1. Definimos las rutas de prueba (usando rutas absolutas)
rutas_img_prueba = [
    os.path.join(project_root, "data/training/images/21_training.tif"), 
    os.path.join(project_root, "data/training/images/22_training.tif")
]
rutas_mask_prueba = [
    os.path.join(project_root, "data/training/1st_manual/21_manual1.gif"), 
    os.path.join(project_root, "data/training/1st_manual/22_manual1.gif")
]

print("Inicializando el motor de datos...")
# 2. Instanciamos tu generador
generador_prueba = DataGenerator(
    rutas_imagenes=rutas_img_prueba, 
    rutas_mascaras=rutas_mask_prueba, 
    batch_size=2, 
    patch_size=(128, 128)
)

print("Construyendo la red neuronal de Lorenzo...")
# 3. Construimos el modelo (Ojo: TensorFlow puede soltar avisos en la terminal aquí, es normal)
modelo_prueba = construir_unet(input_shape=(128, 128, 3))

print("¡Extrayendo lote y haciendo predicción!")
# 4. LA PRUEBA DE FUEGO
X_batch, y_batch = generador_prueba[0]
predicciones = modelo_prueba.predict(X_batch)

print("-" * 40)
print("TEST COMPLETADO. RESULTADOS MATEMÁTICOS:")
print(f"Dimensiones de entrada (X): {X_batch.shape}")
print(f"Dimensiones esperadas (y):  {y_batch.shape}")
print(f"Dimensiones de salida:      {predicciones.shape}")
print("-" * 40)

# 5. Visualizamos el resultado
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(X_batch[0])
axs[0].set_title("1. Entrada (Lo que ve la IA)")
axs[0].axis('off')

axs[1].imshow(y_batch[0, :, :, 0], cmap='gray')
axs[1].set_title("2. Ground Truth (Lo que debería ser)")
axs[1].axis('off')

axs[2].imshow(predicciones[0, :, :, 0], cmap='gray')
axs[2].set_title("3. Predicción (Ruido inicial)")
axs[2].axis('off')

plt.show()