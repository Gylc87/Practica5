import cv2
import numpy as np


img = cv2.imread("bookpage.jpg")

grises =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retval, umbral = cv2.threshold( grises, 10, 255, cv2.THRESH_BINARY)
retval2, umbral2 = cv2.threshold( grises, 125, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
umbralAdaptive = cv2.adaptiveThreshold(grises, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 155, 1)
cv2.imshow("Original",img)
cv2.imshow("Umbral", umbral)
cv2.imshow("Umbral adaptativo", umbralAdaptive)
cv2.imshow("Umbral Otsu", umbral2)

cv2.waitKey(0)
cv2.destroyAllWindows()