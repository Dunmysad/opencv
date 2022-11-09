import cv2
import os

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
sampleNum = 0
datapath = 'dataSet'

def join():
	global sampleNum, imgGray
	id, name = input('enter user id and name: ').split(' ')
	while True:
		sucess, img = cap.read()
		imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(imgGray,1.1,4)
		for (x, y, w, h) in faces:
			sampleNum = sampleNum + 1
			if not os.path.exists(f'{datapath}/{name}'):
				os.makedirs(f'{datapath}/{name}')
			cv2.imwrite(f'{datapath}/{name}/{str(id)}.{str(sampleNum)}.jpg', imgGray[y:y+h, x:x+w])
			cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
			cv2.waitKey(100)
			cv2.imshow('img', imgGray[y:y+h, x:x+w])

			print(f'{datapath}/{name}/{str(id)}.{str(sampleNum)}.jpg')	
			
		cv2.waitKey(1)
		if (sampleNum > 30):
			break
	print(name)
	return name

if __name__ == '__main__':
	join()
