import cv2
import numpy as np
import os
from datetime import datetime


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)


def returnName():
    path = 'dataSet'
    names = {}
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    for i in image_paths:
        Name = i.split('/')[1] # name
        Id = os.listdir(i)[0].split('.')[0] # id
        names[Id] = Name
    # print(names, id)
    return names

def main():
    cam = cv2.VideoCapture(0)

    cam.set(10,100)
    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.1 ,5)
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            if(conf<50):
                if(Id not in names):
                    Id = names[str(Id)]

            else:
                Id ="Unknown"
                
            cv2.putText(im , str(Id), (x,y), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0) , 3)
        cv2.imshow('im',im) 
        if cv2.waitKey(10) & 0xFF==ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    names = returnName()
    main()