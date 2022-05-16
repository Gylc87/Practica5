from random import gauss
import cv2
import numpy as np
from matplotlib import pyplot as plt



img = cv2.imread("bookpage.jpg")

retval, umbral = cv2.threshold( img, 12, 255, cv2.THRESH_BINARY)

grises =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retval, umbral2 = cv2.threshold( grises, 12, 255, cv2.THRESH_BINARY)
Gaus = cv2.adaptiveThreshold(grises, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
retval, otsu = cv2.threshold( grises, 125, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
retval, umbral3 = cv2.threshold( img, 127, 255, cv2.THRESH_BINARY_INV)


titles = ["Origianl","BINARY", "BINARY GRISES", "GAUS", "OTSU", "BINARY INV"]
imagenes = [img,umbral,umbral2,Gaus,otsu,umbral3]

ArrayImg = np.arange(6)
for i in ArrayImg:
    plt.subplot(2,3,i +1),plt.imshow(imagenes[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
""" cv2.imshow("Original",img)
cv2.imshow("Umbral THRESH_BINARY", umbral)
cv2.imshow("Umbral THRESH_BINARY GRISES", umbral2)
cv2.imshow("Umbral Gaus", Gaus)
cv2.imshow("Umbral Otsu", otsu)
cv2.imshow("Umbral THRESH_BINARY_INV", umbral3) 

cv2.waitKey(0)
cv2.destroyAllWindows()"""