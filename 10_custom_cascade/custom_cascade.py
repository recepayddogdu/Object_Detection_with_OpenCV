"""
**Yapacagimiz islemler;**

1- Veriseti olustur.
    - Verisetinin icinde negatif ve pozitif goruntuler olacak.
        - Pozitif: Tespit etmek istedigimiz objeyi iceren goruntuler.
        - Negatif: Tespit edilecek objeyi icermeyen guruntuler.

2- Cascade programi indirilecek.

3- Cascade olusturulacak.

4- Cascade ile tespit algoritmasi yazilacak.
"""

import cv2
import os

# Veriseti depo klasoru
path = "images"

# Goruntu boyutu
imgWidth = 180 #Genislik
imgHeight = 120 #Yukseklik

#Video capture
cap = cv2.VideoCapture(1)

cap.set(3, 640) #Width
cap.set(4, 480) #Height
cap.set(10, 180) #Brightness
print("fps : " + str(cap.get(5)))

global countFolder
def saveDataFunc():
    global countFolder
    countFolder = 0

    while os.path.exists(path + str(countFolder)):
        countFolder += 1
    # os.makedirs(path + str(countFolder))
    
saveDataFunc()

count = 0     # 5 frame'de bir frame kontrolu icin
countSave = 0 # frame'lerin ismi icin

while True:
    success, img = cap.read()
    
    if success:
        cv2.imshow("Image", img)
        
        img = cv2.resize(img, (imgWidth, imgHeight))
        
        #Her frame'i almaya gerek olmadigi icin 5'in katlarina denk gelen frame'ler alinacak.
        if count%5==0:
            # cv2.imwrite(path + str(countFolder) + "/" + str(countSave) + "_.png", img)
            countSave += 1
            print(path + str(countFolder) + "/" + str(countSave) + "_.png")
        
        count += 1
    
        # cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    