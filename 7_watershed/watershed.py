import cv2
import matplotlib.pyplot as plt
import numpy as np

#gorsellestirme fonksiyonu
def imshow_img(img, title):
    plt.figure(), plt.imshow(img, cmap="gray"), plt.title(title)

coin_color = cv2.imread("7_watershed\coins.jpg")
coin = cv2.cvtColor(coin_color, cv2.COLOR_BGR2GRAY)
imshow_img(coin, "original")

#Low pass filter: Blurring
coin_blur = cv2.medianBlur(coin, ksize=13)
imshow_img(coin_blur, "coin_blur")

# binary threshold
ret, coin_thresh = cv2.threshold(coin_blur, 75, 255, cv2.THRESH_BINARY)
imshow_img(coin_thresh, "binary threshold w blur")

ret, coin_thresh_not_blur = cv2.threshold(coin, 75, 255, cv2.THRESH_BINARY)
imshow_img(coin_thresh_not_blur, "binary threshold")

# contour
contours, hierarchy = cv2.findContours(coin_thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1: #external
        cv2.drawContours(coin_color, contours, i, (0,255,0), 10)
plt.figure(), plt.imshow(coin_color), plt.title("contours")


### WATERSHED

coin_color = cv2.imread("7_watershed\coins.jpg")
coin = cv2.cvtColor(coin_color, cv2.COLOR_BGR2GRAY)
#imshow_img(coin, "original")

#Low pass filter: Blurring
coin_blur = cv2.medianBlur(coin, ksize=13)
# imshow_img(coin_blur, "coin_blur")

# binary threshold
ret, coin_thresh = cv2.threshold(coin_blur, 75, 255, cv2.THRESH_BINARY)
imshow_img(coin_thresh, "binary threshold w blur 90")

ret, coin_thresh_not_blur = cv2.threshold(coin, 75, 255, cv2.THRESH_BINARY)
# imshow_img(coin_thresh_not_blur, "binary threshold")

# opening
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(coin_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
imshow_img(opening, "opening")

# nesneler arasi distance bulma
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
imshow_img(dist_transform, "dist_transform")

#resmi kucult
ret, sure_foreground = cv2.threshold(dist_transform, 0.4*np.max(dist_transform),255,0)
sure_foreground = np.uint8(sure_foreground)
imshow_img(sure_foreground, "kucultme")

#arka plan icin resmi buyut
sure_background = cv2.dilate(opening, kernel, iterations=1)
sure_background = np.uint8(sure_background)

#arkaplan-onplan arasindaki fark
unknown = cv2.subtract(sure_background, sure_foreground)
imshow_img(unknown, "unknown")

# baglanti
ret, marker = cv2.connectedComponents(sure_foreground)
marker = marker + 1
marker[unknown==255] = 0
imshow_img(marker, "marker")

#watershed
marker = cv2.watershed(coin_color, marker)
imshow_img(marker, "watershed")

# contour
contours, hierarchy = cv2.findContours(marker.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    if hierarchy[0][i][3] == -1: #external
        cv2.drawContours(coin_color, contours, i, (0,255,0), 10)
plt.figure(), plt.imshow(coin_color), plt.title("contours")


plt.show()