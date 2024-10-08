import cv2
import numpy as np
from skimage.exposure import match_histograms

def upsample(image, scale_factor):
    # Obtener dimensiones de la imagen original
    h, w = image.shape
    
    # Crear una nueva imagen de tamaño aumentado
    new_h = h * scale_factor
    new_w = w * scale_factor
    upsampled_image = np.zeros((new_h, new_w))

    # Asignar los píxeles de la imagen original a la nueva imagen
    for i in range(h):
        for j in range(w):
            upsampled_image[i * scale_factor, j * scale_factor] = image[i, j]

    # Crear un kernel de suavizado para la interpolación
    kernel = np.ones((scale_factor, scale_factor)) / (scale_factor ** 2)

    # Aplicar la convolución usando filter2D
    upsampled_image = cv2.filter2D(upsampled_image, -1, kernel)

    # Normalizar la imagen resultante
    normalized_image = cv2.normalize(upsampled_image, None, 0, 255, cv2.NORM_MINMAX)

    # Aplicamos la LUT de la imagen original a la aumentada para que mantenga los mismos colores
    out_image = match_histograms(normalized_image, image)
    
    return out_image

if __name__ == "__main__":
    image_path = 'images/input/lena.jpeg'
    save_path = 'images/output/lena.jpeg'
    
    image = cv2.imread(image_path, 0)

    out_image = upsample(image, scale_factor=4) 

    # Guardar la imagen resultante
    cv2.imwrite(save_path, out_image)