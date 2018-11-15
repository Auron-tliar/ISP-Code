import dlib
import numpy as np
from DetectorResult import DetectorResult
from Trainer import DETECTOR_SVM
from Trainer import PREDICTOR_DAT
from skimage import io
import cv2 as cv


class Detector:
    def __init__(self, input_image):
        self.input_image = input_image
        self.image_data = io.imread(input_image)
        # self.detector = dlib.fhog_object_detector(DETECTOR_SVM)
        self.predictor = dlib.shape_predictor(PREDICTOR_DAT)
        self.result = DetectorResult()

    def detect(self):
        self.result.faces = self.detector(self.image_data, 1)
        # self.result.faces = self.FaceDetectionLBP(self.image_data)
        self.result.face_count = len(self.result.faces)


    def FaceDetectionLBP(self, img):
        face_cascade = cv.CascadeClassifier("visionary.net_cat_cascade_web_LBP.xml")
        # face_cascade = cv.CascadeClassifier("haarcascade_frontalcatface.xml")
        # face_cascade = cv.CascadeClassifier("lbpcascade_frontalcatface.xml")

        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        scale = 1.1
        neighbors = 4

        # while True:
        faces = face_cascade.detectMultiScale(img, scale, neighbors)
        # if len(faces) == 1:
        #     break
        # elif len(faces) == 0:
        #     if scale > 1.01:
        #         scale -= 0.01
        #     else:
        #         return False
        # else:
        #     if neighbors <= 15:
        #         neighbors += 1
        #     else:
        #         faces = [faces[0]]
        #         break
        output = []

        for face in faces:
            (x, y, w, h) = face
            # output.append(img[y:y + h, x:x + w])
            output.append(dlib.rectangle(x, y, x+w, y+h))
        return output

        # for (x,y,w,h) in faces:
        #     cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        # cv.imshow('img',img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
