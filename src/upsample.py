import cv2
import numpy as np
from skimage.exposure import match_histograms

def upsample_with_filter2D(image, scale_factor):
    # Obtener dimensiones de la imagen original
    h, w, c = image.shape
    
    # Crear una nueva imagen de tamaño aumentado
    new_h = h * scale_factor
    new_w = w * scale_factor
    upsampled_image = np.zeros((new_h, new_w, c), dtype=np.float32)

    # Asignar los píxeles de la imagen original a la nueva imagen
    for i in range(h):
        for j in range(w):
            upsampled_image[i * scale_factor, j * scale_factor] = image[i, j]

    # Crear un kernel de suavizado para la interpolación
    kernel = np.ones((scale_factor, scale_factor), np.float32) / (scale_factor ** 2)

    # Aplicar la convolución usando filter2D
    upsampled_image = cv2.filter2D(upsampled_image, -1, kernel)

    # Normalizar la imagen resultante
    normalized_image = cv2.normalize(upsampled_image, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized_image

if __name__ == "__main__":
    # Ruta de la imagen
    image_path = 'images/input/lena.jpeg'
    save_path = 'images/output/lena.jpeg'
    
    # Cargar la imagen
    original_image = cv2.imread(image_path)

    # Aplicar la interpolación usando filter2D
    upsampled_image = upsample_with_filter2D(original_image, scale_factor=2) 

    # Aplicar la LUT a la imagen interpolada
    lut_applied_image = match_histograms(upsampled_image, original_image)

    # Guardar la imagen resultante
    cv2.imwrite(save_path, lut_applied_image)