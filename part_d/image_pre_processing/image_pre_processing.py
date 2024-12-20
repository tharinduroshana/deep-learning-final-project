import cv2
import numpy as np
from PIL import Image


def ben_graham_preprocessing(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blurred_image = cv2.GaussianBlur(img, (0, 0), sigmaX=10)
    processed_image = cv2.addWeighted(img, 4, blurred_image, -4, 128)
    return Image.fromarray(processed_image)

def circle_cropping(image):
    img = np.array(image)
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    center = (h // 2, w // 2)
    radius = min(center)
    mask = (x - center[1]) ** 2 + (y - center[0]) ** 2 <= radius ** 2
    circular_img = np.zeros_like(img)
    circular_img[mask] = img[mask]
    return Image.fromarray(circular_img)

def clahe_preprocessing(image):
    img = np.array(image)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    return Image.fromarray(cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB))

def gaussian_blur(image, kernel_size=5):
    img = np.array(image)
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 10)
    return Image.fromarray(blurred_img)

def sharpen_image(image):
    img = np.array(image)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_img = cv2.filter2D(img, -1, kernel)
    return Image.fromarray(sharpened_img)
