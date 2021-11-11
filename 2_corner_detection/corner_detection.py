import cv2
import matplotlib.pyplot as plt
import numpy as np

#gorsellestirme fonksiyonu
def imshow_img(img, title):
    plt.figure(), plt.imshow(img, cmap="gray"), plt.title(title)

img = cv2.imread("2_corner_detection/sudoku.jpg", 0)
img = np.float32(img) #ondalikli sayilara cevirme
print(img.shape)
imshow_img(img, "original")

#harris corner detection
dst = cv2.cornerHarris(img, blockSize=2, ksize=3, k=0.04)
imshow_img(dst, "cornerHarris")

#dilate yontemi ile tespit edilen noktalari genisletme
dst = cv2.dilate(dst, None)
img[dst>0.2*dst.max()] = 1 

imshow_img(dst, "cornerHarris")

# shi tomasi detection
img = cv2.imread("2_corner_detection/sudoku.jpg", 0)
img = np.float32(img) #ondalikli sayilara cevirme

corners = cv2.goodFeaturesToTrack(img,
                                    120, #istenilen corner sayisi
                                    0.01, #quality level
                                    10)  #iki kose arasindaki min distance

corners = np.int64(corners)

for i in corners:
    x,y = i.ravel() #duzlestirme
    cv2.circle(img, (x,y),3,(125,125,125), cv2.FILLED)

plt.figure(),plt.imshow(img)

plt.show()