import cv2

path = "classifier/cascade.xml"
objectName = "Mint"
frameWidth = 280
frameHeight = 360
color = (255,0,255)

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# trackbar
cv2.namedWindow("Sonuc")
cv2.resizeWindow("Sonuc", frameWidth, frameHeight + 100)

# trackbar'lar icin bos fonksiyon
def empty(a): pass

#detectMultiscale fonksiyonu icerisindeki scale degerini degistirir.
cv2.createTrackbar("Scale","Sonuc", 500, 1000, empty)


cv2.createTrackbar("Neighbor","Sonuc", 4, 50, empty)

# cascade classifier
cascade = cv2.CascadeClassifier(path)



while True:
    success, img = cap.read()
    
    if success:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detection parametreleri
        # scale normalde 1-2 arasinda olur fakat rahat hareket ettirebilmek icin boyle yaptik.
        scaleVal = 1 + (cv2.getTrackbarPos("Scale", "Sonuc") / 1000)
        
        neighbor = cv2.getTrackbarPos("Neighbor", "Sonuc")
        
        # detection
        rects = cascade.detectMultiScale(gray, scaleVal, neighbor)
        
        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 3)
            cv2.putText(img, objectName, (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
        
        cv2.imshow("Sonuc", img)
        
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        