import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from math import sqrt

def test():
    R = np.array((181, 172, 179, 179, 172, 181, 177, 185, 185), dtype=np.float32).reshape(3, 3)
    G = np.array((71, 68, 70, 70, 68, 71, 62, 74, 74), dtype=np.float32).reshape(3, 3)
    B = np.array((81, 76, 79, 79, 76, 81, 79, 81, 81), dtype=np.float32).reshape(3, 3)
    image = cv2.merge((R, G, B))
    return image

def calc_mse(original_image, stimated_image):
    """
      Calculamos el Error Cuadratico Medio de Acuerdo entre dos matrices.
    """
    diff = original_image - stimated_image
    
    mse = np.mean(diff**2)
    return mse

  
def calc_psnr(original_image, stimated_image):
    """
      Calculamos el Proporción Máxima de Señal a Ruido dado un mse
    """
    
    mse = calc_mse(original_image, stimated_image)
    psnr = sqrt(mse)
    return psnr

  
def calc_mae(original_image, stimated_image):
    """
      Calculamos el Error absoluto medio de Acuerdo entre dos matrices.
    """
    diff = original_image - stimated_image
    mae = np.mean(np.abs(diff))
    return mae

def add_salt_and_pepper(image, prob):
    rnd = np.random.rand(image.shape[0], image.shape[1])
    noise = image.copy()
    noise[rnd < prob] = (0, 0, 0)
    noise[rnd > 1 - prob] = [255, 255, 255]
    return noise

def vector_median_filter(image, median_window_size):
    height, width, _ = image.shape
    padded_image = cv2.copyMakeBorder(image, median_window_size//2, median_window_size//2, median_window_size//2, median_window_size//2, cv2.BORDER_REFLECT)

    result = np.zeros_like(image, dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            window = padded_image[i:i+median_window_size, j:j+median_window_size, :]
            window = window.reshape((-1, 3))
            window_sorted = np.sort(window, axis=0)
            median = window_sorted[window_sorted.shape[0] // 2]
            result[i, j] = median

    return cv2.cvtColor(result, cv2.COLOR_Lab2LBGR)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True, help='Ubicación de la imagen')
    ap.add_argument('-p', '--prob_salt_pepper', type=float, default=0.05, help='Probabilidad de ruido sal y pimienta')
    ap.add_argument('-w', '--median_window_size', type=int, default=3, help='Tamaño de la ventana de mediana')
    args = vars(ap.parse_args())

    image_url = args['image']
    prob_salt_pepper = args['prob_salt_pepper']
    median_window_size = args['median_window_size']

    image = cv2.imread(image_url)
    
    
    image_noise = add_salt_and_pepper(image, prob_salt_pepper)
    img_lab = cv2.cvtColor(image_noise, cv2.COLOR_RGB2LAB)
    image_processed = vector_median_filter(img_lab, median_window_size)
    image_noise = cv2.cvtColor(image_noise, cv2.COLOR_RGB2BGR)

    plt.subplot(1, 2, 1), plt.imshow(image_noise, vmin=0, vmax=255), plt.title("Imagen Contaminada")
    plt.subplot(1, 2, 2), plt.imshow(image_processed, vmin=0, vmax=255), plt.title("Imagen Resultante")
    plt.show()
    
    mse_original = calc_mse(image, image_processed)
    mae_original = calc_mae(image, image_processed)
    psnr_original = calc_psnr(image, image_processed)
    
    print("Medidas de desempeño")
    
    print("MSE: "+str(mse_original))
    
    print("MAE: "+str(mae_original))
    
    print("PSNR: "+str(psnr_original))
    