import numpy as np
import cv2 as cv
from os import listdir
from os.path import isfile, join
from catfd.catfd import detect
from catfd.lib.Trainer import DETECTOR_SVM

def FaceDetectionLBP(filename):
    face_cascade = cv.CascadeClassifier("visionary.net_cat_cascade_web_LBP.xml")
    # face_cascade = cv.CascadeClassifier("haarcascade_frontalcatface.xml")
    # face_cascade = cv.CascadeClassifier("lbpcascade_frontalcatface.xml")

    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    scale = 1.1
    neighbors = 4

    while True:
        faces = face_cascade.detectMultiScale(img, scale, neighbors)
        if len(faces) == 1:
            break
        elif len(faces) == 0:
            if scale > 1.01:
                scale -= 0.01
            else:
                return False
        else:
            if neighbors <= 15:
                neighbors += 1
            else:
                faces = [faces[0]]
                break

    (x, y, w, h) = faces[0]
    return img[y:y+h, x:x+w]

    # for (x,y,w,h) in faces:
    #     cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    # cv.imshow('img',img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()



if __name__ == '__main__':
    # onlyfiles = [f for f in listdir("same-cats") if isfile(join("same-cats", f))]
    # # print onlyfiles
    # # fs = [onlyfiles[0]]
    # for f in onlyfiles:
    #     cv.imshow('img', FaceDetectionLBP(join("same-cats", f)))
    #     cv.waitKey(0)
    # cv.destroyAllWindows()
    detect('a1.jpg', '.', False, True, True, [25, 255, 100], [255, 50, 100], True)