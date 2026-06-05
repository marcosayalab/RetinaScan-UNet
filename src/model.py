import tensorflow as tf
from tensorflow.keras import layers, models

def bloque_convolucion_doble(x, num_filtros):
    
    # Primera convolución
    x = layers.Conv2D(num_filtros, kernel_size=3, padding="same", 
                      activation="relu", kernel_initializer="he_normal")(x)
    # Segunda convolución
    x = layers.Conv2D(num_filtros, kernel_size=3, padding="same", 
                      activation="relu", kernel_initializer="he_normal")(x)
    return x

def construir_unet(input_shape=(128, 128, 3)):

    inputs = layers.Input(shape=input_shape)
    
    # 1. RUTA DE CONTRACCIÓN (ENCODER)
    
    # Bloque 1
    c1 = bloque_convolucion_doble(inputs, 16)
    p1 = layers.MaxPooling2D(pool_size=(2, 2))(c1)
    
    # Bloque 2
    c2 = bloque_convolucion_doble(p1, 32)
    p2 = layers.MaxPooling2D(pool_size=(2, 2))(c2)
    
    # Bloque 3
    c3 = bloque_convolucion_doble(p2, 64)
    p3 = layers.MaxPooling2D(pool_size=(2, 2))(c3)
    
    # Bloque 4
    c4 = bloque_convolucion_doble(p3, 128)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)
    
    # 2. CUELLO DE BOTELLA (BOTTLENECK) (Parte más profunda de la U-Net)

    c5 = bloque_convolucion_doble(p4, 256)
    
    # 3. RUTA DE EXPANSIÓN (DECODER)

    # Bloque 6 (Sube de nivel desde c5 y concatena con c4)
    u6 = layers.Conv2DTranspose(128, kernel_size=2, strides=(2, 2), padding="same")(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = bloque_convolucion_doble(u6, 128)
    
    # Bloque 7
    u7 = layers.Conv2DTranspose(64, kernel_size=2, strides=(2, 2), padding="same")(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = bloque_convolucion_doble(u7, 64)
    
    # Bloque 8
    u8 = layers.Conv2DTranspose(32, kernel_size=2, strides=(2, 2), padding="same")(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = bloque_convolucion_doble(u8, 32)
    
    # Bloque 9
    u9 = layers.Conv2DTranspose(16, kernel_size=2, strides=(2, 2), padding="same")(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = bloque_convolucion_doble(u9, 16)
    
    # 4. CAPA DE SALIDA

    # Una sola neurona/filtro con activación Sigmoide para segmentación binaria
    outputs = layers.Conv2D(1, kernel_size=1, padding="same", activation="sigmoid")(c9)
    
    model = models.Model(inputs=[inputs], outputs=[outputs], name="UNet_Segmentacion_Medica")
    return model

# Inicializar y comprobar el modelo:
modelo_unet = construir_unet(input_shape=(128, 128, 3))
modelo_unet.summary()