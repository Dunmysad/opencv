import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

id = input('enter user id: ')
sampleNum = 0
while True:
	sucess, img = cap.read()
	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(imgGray,1.1,4)
	
	for (x, y, w, h) in faces:
		sampleNum = sampleNum + 1
		cv2.imwrite('dataSet/' + str(id) + '.' + str(sampleNum) + '.jpg', imgGray[y:y+h, x:x+w])
		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
		cv2.waitKey(100)
		cv2.imshow('img', imgGray[y:y+h, x:x+w])
	
	cv2.waitKey(1)
	if (sampleNum > 20):
		break
