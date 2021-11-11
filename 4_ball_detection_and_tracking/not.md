c= max..... yerine,

    for c in contours:

        rect.....



yazarak aynı işlemleri yapabilirsin. o zaman tüm mavi nesneleri tespit eder ve kutu içine alır. ama gürültülerde tespit edilmiş olur ve güzel bir görüntü olmaz. bunu engellemek için for dan sonra bir koşul koyarak alanı belli bir değerin üzerinde olanları çizdirirsen daha iyi olabilir. mesela:

    for c in contours:

        if cv2.contourArea(c) > 500:

            rect.......

### Skin Tone HSV
0, 10, 60
20, 150, 255