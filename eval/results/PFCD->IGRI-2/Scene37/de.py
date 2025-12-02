import cv2
import numpy as np

img = cv2.imread('Dem_0_ori.png').astype(np.uint16)

img = np.clip(img + 1, 0, 255).astype(np.uint8)


cv2.imwrite('Dem_0.png', img)