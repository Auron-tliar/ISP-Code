import numpy as np
import cv2 as cv
import argparse
import glob
import os
from os import listdir
from os.path import isfile, join
from catfd.catfd import detect, get_landmarks, get_face
from catfd.lib.Trainer import DETECTOR_SVM
from openface import AlignDlib
from PIL import Image
from dlib import rectangle
from imutils import face_utils


def FaceDetectionLBP(img):
    face_cascade = cv.CascadeClassifier("visionary.net_cat_cascade_web_LBP.xml")
    # face_cascade = cv.CascadeClassifier("haarcascade_frontalcatface.xml")
    # face_cascade = cv.CascadeClassifier("lbpcascade_frontalcatface.xml")
    (H,W,D) = img.shape
    imgG = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    scale = 1.1
    neighbors = 4

    while True:
        faces = face_cascade.detectMultiScale(imgG, scale, neighbors)
        if len(faces) == 1:
            break
        elif len(faces) == 0:
            if scale > 1.01:
                scale -= 0.01
            else:
                return None
        else:
            bestInd = -1
            bestVal = 0
            for i in range(len(faces)):
                (x, y, w, h) = faces[i]
                if w*h > bestVal:
                    bestInd = i
                    bestVal = w*h
            faces[0] = faces[bestInd]
            break
            # if neighbors <= 15:
            #     neighbors += 1
            # else:
            #     faces = [faces[0]]
            #     break

    (x, y, w, h) = faces[0]
    wmod = int(0.1 * w)
    hmod = int(0.2 * h)
    x = max(0,x-wmod)
    y = max(0,y-2*hmod)
    if x + w + wmod > W:
        w = W-1
    else:
        w += wmod
    if y + h + hmod > H:
        h = H-1
    else:
        h += hmod
    return (x, y, w, h)#img[y:y+h, x:x+w]

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

    formatter = lambda prog: argparse.HelpFormatter(prog,
                                                    max_help_position=36)
    desc = '''
    Detects cat faces and facial landmarks
    '''
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=formatter)

    parser.add_argument('-i', '--input-image',
                        help='input image',
                        metavar='<file>')

    parser.add_argument('-f', '--input-folder',
                        help='input folder',
                        metavar='<path>')

    args = vars(parser.parse_args())

    if not args['input_image'] and not args['input_folder']:
        parser.error("must specify either -i or -f")

    if args['input_image']:
        print("OUT OF ORDER")
        # detect(args['input_image'], './Temp/', False, True, True, [25, 255, 100], [255, 50, 100], True)
    if args['input_folder']:
        for f in glob.glob(os.path.join(args['input_folder'], '*.jp*g')):
            print(f)
            img = cv.imread(f)
            # cv.imwrite(f + "_flip.jpg", cv.flip(img,1))
            face = FaceDetectionLBP(img)
            # face = get_face(f, img)
            if face is not None:
                # cv.imwrite(f + "_test.jpg", cv.cvtColor(img, cv.COLOR_RGB2BGR))
                (x,y,w,h) = face
                rect = rectangle(left=x, top=y, right=x+w, bottom=y+h)
                cropped = img[y:y + h, x:x + w]
                cv.imwrite(f + "_cropped.jpg", cropped)
                # detect(f, './Temp/', False, True, True, [25, 255, 100], [255, 255, 0], True)
                shape = get_landmarks(f, img, rect)
                shape = face_utils.shape_to_np(shape)
                align = AlignDlib("")
                aligned = align.align(96, img, bb=rect,
                                          landmarks=shape)
                cv.imwrite(f + "_aligned.jpg", aligned)
            else:
                print("No face detected!")