import cv2
import numpy as np


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

cam = cv2.VideoCapture(0)

cam.set(10,100)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf<50):
            if(Id == 1):
                Id="lishihao"
            elif(Id == 2):
                Id ='chenshuai'
        else:
            Id="Unknown"
            
        cv2.putText(im ,str(Id) + '~' + str(conf), (x,y+h), cv2.FONT_HERSHEY_COMPLEX, 3, (0,255,0) , 3)
    cv2.imshow('im',im) 
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
