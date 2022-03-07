import cv2
import matplotlib.pyplot as plt
import numpy as np

#gorsellestirme fonksiyonu
def imshow_img(img, title):
    plt.figure(), plt.imshow(img, cmap="gray"), plt.title(title)

img = cv2.imread("1_edge_detection/london.jpg", 0)
imshow_img(img, "original")

edges = cv2.Canny(image=img, threshold1=0, threshold2=255)
imshow_img(edges, "edges")

med_val = np.median(img)
print(med_val)

low = int(max(0, (1-0.33)*med_val)) # =85
high = int(min(255, (1+0.33)*med_val)) # =169

edges_th = cv2.Canny(image=img, threshold1=low, threshold2=high)
imshow_img(edges_th, "edges_th (85, 169)")

# blur
blurred_img = cv2.blur(img, ksize=(7,7))
imshow_img(blurred_img, "blurred img")

med_val = np.median(blurred_img)
print(med_val)

low = int(max(0, (1-0.33)*med_val)) # =85
high = int(min(255, (1+0.33)*med_val)) # =169

edges_th = cv2.Canny(image=blurred_img, threshold1=low, threshold2=high)
imshow_img(edges_th, "edges_th - blurred (85, 169)")

plt.show()