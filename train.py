import cv2
import os
import numpy as np
from PIL import Image

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()


def get_images_and_labels(path):

    face_samples = []
    ids = []
    names = []

    for image_path in path:
        image = Image.open(image_path).convert('L')
        image_np = np.array(image, 'uint8')
        if os.path.split(image_path)[-1].split(".")[-1] != 'jpg':
            continue
        image_name = int(os.path.split(image_path)[-1].split(".")[1])
        image_id = int(os.path.split(image_path)[-1].split(".")[0])
        # print(image_id, image_name)
        faces = detector.detectMultiScale(image_np)
        for (x, y, w, h) in faces:
            face_samples.append(image_np[y:y + h, x:x + w])
            ids.append(image_id)
            names.append(image_name)

    return face_samples, ids

if __name__ == '__main__':
    path = 'dataSet'
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    print(image_paths)
    path = []
    for i in image_paths:
        for l in os.listdir(i):
            # print(f'{i}/{l}')
            path.append(f'{i}/{l}')

    faces, Ids = get_images_and_labels(path)
    recognizer.train(faces, np.array(Ids))
    recognizer.write('trainner.yml')
    print('训练完成')
    

