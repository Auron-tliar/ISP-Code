import numpy as np
import cv2 as cv
import argparse
import glob
import os
from scipy.stats import ttest_ind
import pickle
from openface import AlignDlib
import dlib
from dlib import rectangle
from imutils import face_utils


FOLDER = 'classes'
ALPHA = 0.1
PREDICTOR_DAT = 'predictor.dat'


# Detect face box in image
def FaceDetectionLBP(img):
    face_cascade = cv.CascadeClassifier("visionary.net_cat_cascade_web_LBP.xml")
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
    return (x, y, w, h)


# Calculate inner similarities of the class
def InnerSimilarity(faces):
    hists = []

    for face in faces:
        hist = cv.calcHist([face], [0,1,2], None, [8,8,8], [0, 256, 0, 256, 0, 256])
        hist = cv.normalize(hist, hist).flatten()
        hists.append(hist)

    sims = []
    N = len(hists)
    for i in range(N):
        for j in range(N):
            if i != j:
                sims.append(cv.compareHist(hists[i], hists[j], cv.HISTCMP_CORREL))

    return sims


# calculate similarities between face and members of a set of faces
def CalculateSimilarities(face, faces):
    hist = cv.calcHist([face], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv.normalize(hist, hist).flatten()

    sims = []
    for f in faces:
        h = cv.calcHist([f], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        h = cv.normalize(h, h).flatten()
        sims.append(cv.compareHist(hist, h, cv.HISTCMP_CORREL))

    return sims


#train face recognizer based on LBPH
def TrainFaceRecognizer(folder):
    labels = []
    faces = []
    if folder is not None:
        dirs = os.listdir(folder)
        for dir in dirs:
            curFaces = []
            files = glob.glob(folder + '/' + dir + '/*.jp*g')
            for file in files:
                faces.append(cv.imread(file))
                curFaces.append(faces[-1])
                labels.append(int(dir))
    
            with open(dir + '.cim', 'wb') as file:
                pickle.dump(curFaces, file)
            with open(dir + '.csm', 'wb') as file:
                pickle.dump(InnerSimilarity(curFaces), file)
    else:
        files = glob.glob('*.cim')
        for file in files:
            with open(file, 'rb') as f:
                curFaces = pickle.load(f)
                faces.extend(curFaces)
                for i in range(len(curFaces)):
                    labels.append(int(os.path.splitext(file)[0]))
    for i in range(len(faces)):
        faces[i] = cv.cvtColor(faces[i], cv.COLOR_BGR2GRAY)

    faces = np.array(faces)
    labels = np.array(labels)
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, labels)

    face_recognizer.write('lbph_face_recognizer')
    return face_recognizer


# check if a face is similar enough to members of a class using t-test
def CheckSimilarity(face, faces, classSimilarities):
    sims = CalculateSimilarities(face, faces)

    stat, p = ttest_ind(sims, classSimilarities)
    if p > ALPHA:
        return True
    else:
        return False


# extract aligned face from image
def ExtractFace(img):
    face = FaceDetectionLBP(img)
    if face is not None:
        (x, y, w, h) = face
        rect = rectangle(left=x, top=y, right=x + w, bottom=y + h)
        shape = GetLandmarks(img, rect)
        shape = face_utils.shape_to_np(shape)
        align = AlignDlib("")
        aligned = align.align(96, img, bb=rect,
                              landmarks=shape)

        return aligned
    else:
        return None


# recognize face
def RecognizeFace(face):
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read('lbph_face_recognizer')
    label = face_recognizer.predict(face)
    return label


# find landmarks of the face
def GetLandmarks(img, face):
    predictor = dlib.shape_predictor(PREDICTOR_DAT)
    shape = predictor(img, face)
    return shape


# check whether the image contains a face of one of the existing classes
# THIS IS THE MAIN FUNCTION TO CALL
def CheckImage(img):
    face = ExtractFace(img)
    if face is not None:
        label = RecognizeFace(cv.cvtColor(face, cv.COLOR_BGR2GRAY))
        label = label[0]
        with open(str(label) + '.cim', 'rb') as fim:
            with open(str(label) + '.csm', 'rb') as fsim:
                result = CheckSimilarity(face, pickle.load(fim), pickle.load(fsim))

        if result:
            return label
        else:
            return -1
    else:
        return -2


# adds a new class
def AddClass(images, label):
    faces = []
    for img in images:
        face = ExtractFace(img)
        if face is not None:
            faces.append(face)
    with open(str(label) + '.cim', 'wb') as file:
        pickle.dump(faces, file)
    with open(str(label) + '.csm', 'wb') as file:
        pickle.dump(InnerSimilarity(faces), file)


# modifies an existing class
def ModifyClass(img, label):
    face = ExtractFace(img)
    if face is not None:
        with open(str(label) + 'cim', 'rb') as file:
            faces = pickle.load(file)

        with open(str(label) + 'csm', 'rb') as file:
            sims = pickle.load(file)
        sims.extend(CalculateSimilarities(face, faces))
        with open(str(label) + 'csm', 'wb') as file:
            pickle.dump(sims, file)

        faces.append(face)
        with open(str(label) + 'cim', 'wb') as file:
            pickle.dump(faces, file)


if __name__ == '__main__':
    formatter = lambda prog: argparse.HelpFormatter(prog,
                                                    max_help_position=36)
    desc = '''
    Detects cat faces and facial landmarks
    '''
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=formatter)

    parser.add_argument('-c', '--check-image',
                        help='input image',
                        metavar='<file>')

    parser.add_argument('-r', '--retrain', action='store_true',
                        help='start face recognition training')

    args = vars(parser.parse_args())

    if not args['check_image'] and not args['retrain']:
        parser.error("must specify either -r or -c")

    if args['retrain']:
        TrainFaceRecognizer()

    if args['check_image']:
        print(CheckImage(cv.imread(args['check_image'])))

