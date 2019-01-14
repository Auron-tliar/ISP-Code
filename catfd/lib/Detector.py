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
        self.detector = dlib.fhog_object_detector(DETECTOR_SVM)
        self.predictor = dlib.shape_predictor(PREDICTOR_DAT)
        self.result = DetectorResult()

    def detect(self):
        self.result.faces = self.detector(self.image_data, 1)
        # self.result.faces = self.FaceDetectionLBP(self.image_data)
        self.result.face_count = len(self.result.faces)


    def FaceDetectionDLIB(self, img):
        (H, W, D) = img.shape
        imgG = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        upsamples = 1

        while True:
            self.result.faces = self.detector(self.image_data, 1)
            self.result.face_count = len(self.result.faces)
            if self.result.face_count == 1:
                rect = self.result.faces[0]
                break
            elif self.result.face_count == 0:
                if upsamples < 3:
                    upsamples += 2
                else:
                    return None
            else:
                bestInd = -1
                bestVal = 0
                for i in range(self.result.face_count):
                    (x, y, w, h) = self.result.faces[i]
                    if w * h > bestVal:
                        bestInd = i
                        bestVal = w * h
                        self.result.faces[0] = self.result.faces[bestInd]
                rect = self.result.faces[0]
                break
                # if neighbors <= 15:
                #     neighbors += 1
                # else:
                #     faces = [faces[0]]
                #     break

        (x, y, w, h) = (rect.left(), rect.top(), rect.width(), rect.height())
        # wmod = int(0.1 * w)
        # hmod = int(0.2 * h)
        # x = max(0, x - wmod)
        # y = max(0, y - 2 * hmod)
        # if x + w + wmod > W:
        #     w = W - 1
        # else:
        #     w += wmod
        # if y + h + hmod > H:
        #     h = H - 1
        # else:
        #     h += hmod
        return (x, y, w, h)


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
