import cv2
import matplotlib.pyplot as plt
import numpy as np

#gorsellestirme fonksiyonu
def imshow_img(img, title):
    plt.figure(), plt.imshow(img, cmap="gray"), plt.title(title)

img = cv2.imread("3_contour_detection/contour.jpg", 0)
imshow_img(img, "original")

contours, hierarch = cv2.findContours(img,
                                            cv2.RETR_CCOMP, #internal ve external contourler
                                            cv2.CHAIN_APPROX_SIMPLE) #yatay dikey ve capraz bolumleri sıkıstırır,
                                                                    #yanlizca uc noktalarini birakiyor

external_contour = np.zeros(img.shape)
internal_contour = np.zeros(img.shape)

for i in range(len(contours)):
    #external
    if hierarch[0][i][3] == -1:
        cv2.drawContours(external_contour,contours,i,255,-1)
    else:
        cv2.drawContours(internal_contour,contours,i,255,-1)

imshow_img(external_contour, "external contours")  
imshow_img(internal_contour, "internal contours")        
plt.show()