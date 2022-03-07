import cv2
import matplotlib.pyplot as plt

#gorsellestirme fonksiyonu
def imshow_img(img, title):
    plt.figure(), plt.imshow(img, cmap="gray"), plt.title(title)

chos = cv2.imread("6_feature_matching/chocolates.jpg", 0)
imshow_img(chos, "Chocolates")

cho = cv2.imread("6_feature_matching/nestle.jpg", 0)
imshow_img(cho, "nestle")

### Tanimlayicilar

# orb tanimlayicisi
# kose-kenar gibi nesneye ait ozellikler

orb = cv2.ORB_create()

# anahtar nokta tespiti
kp1, des1 = orb.detectAndCompute(cho, None)
kp2, des2 = orb.detectAndCompute(chos, None)

# brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# noktalari eslestir
matches = bf.match(des1, des2)

# mesafeye gore sirala
matches = sorted(matches, key=lambda x: x.distance)

# eslesen resimleri goster

img_match = cv2.drawMatches(cho, kp1, chos, kp2, matches[:20], None, flags=2)
imshow_img(img_match, "match")


# sift
sift = cv2.xfeatures2d.SIFT_create()

# bf
bf = cv2.BFMatcher()

# anahtar nokta tespiti sift ile
kp1, des1 = sift.detectAndCompute(cho, None)
kp2, des2 = sift.detectAndCompute(chos, None)

matches = bf.knnMatch(des1, des2, k = 2)

guzel_eslesme = []

for match1, match2 in matches:
    
    if match1.distance < 0.75*match2.distance:
        guzel_eslesme.append([match1])
    

sift_matches = cv2.drawMatchesKnn(cho,kp1,chos,kp2,guzel_eslesme,None, flags = 2)
imshow_img(sift_matches, "sift")
plt.show()