#!/usr/bin/env python2
#!/usr/bin/python
#
# Example to run classifier on webcam stream.
# Brandon Amos & Vijayenthiran
# 2016/06/21
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Contrib: Vijayenthiran
# This example file shows to run a classifier on webcam stream. You need to
# run the classifier.py to generate classifier with your own dataset.
# To run this file from the openface home dir:
# ./demo/classifier_webcam.py <path-to-your-classifier>


import time

start = time.time()

import argparse
import cv2
import os
import pickle
import sys
import studentStatus as ss
import numpy as np
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
import openface
from ierequests import send_notification
import classifier

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')


def getRep(bgrImg):
    start = time.time()
    if bgrImg is None:
        raise Exception("Unable to load image/frame")

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    start = time.time()

    # Get all bounding boxes
    bb = align.getAllFaceBoundingBoxes(rgbImg)

    if bb is None:
        return None
    start = time.time()

    alignedFaces = []
    for box in bb:
        alignedFaces.append(
            align.align(
                args.imgDim,
                rgbImg,
                box,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))

    if alignedFaces is None:
        raise Exception("Unable to align the frame")
    start = time.time()

    reps = []
    for alignedFace in alignedFaces:
        reps.append(net.forward(alignedFace))

    return reps


def infer(img, args):
    with open(args.classifierModel, 'r') as f:
        if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)  # le - label and clf - classifer
        else:
                (le, clf) = pickle.load(f, encoding='latin1')  # le - label and clf - classifer

    flag = 0
    reps = getRep(img)
    persons = []
    confidences = []
    for rep in reps:
        flag = 1
        try:
            rep = rep.reshape(1, -1)
        except:
            print ("No Face detected")
            return (None, None)
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        persons.append(le.inverse_transform(maxI))
        confidences.append(predictions[maxI])
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))
            pass
    if flag == 0:
        return (persons, confidences)
    else:
        return (person, confidences)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image',
        type=str,
        help="Image to identify",
        default='')
    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument(
        '--networkModel',
        type=str,
        help="Path to Torch network model.",
        default=os.path.join(
            openfaceModelDir,
            'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument(
        '--captureDevice',
        type=int,
        default=2,
        help='Capture device. 2 for regular image from realsense.')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--threshold', type=float, default=0.15)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        'classifierModel',
        type=str,
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')

    args = parser.parse_args()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(
        args.networkModel,
        imgDim=args.imgDim,
        cuda=args.cuda)

    print("\n\n\n------------------------------------------------------------------------------------------\n")
    flag = 0
    if args.image == '':
        # Capture device. For realsense, 0 gives green image, 1 gives grey image, and 2 gives regular RGB image.
        video_capture = cv2.VideoCapture(args.captureDevice)
        video_capture.set(3, args.width)
        video_capture.set(4, args.height)

        confidenceList = []
        while True:
            ret, frame = video_capture.read()
            persons, confidences = infer(frame, args)
            #print ("P: " + str(persons) + " C: " + str(confidences))
            try:
                # append with two floating point precision
                confidenceList.append('%.2f' % confidences[0])
            except:
                # If there is no face detected, confidences matrix will be empty.
                # We can simply ignore it.
                pass

            for i, c in enumerate(confidences):
                for i in range (3):
                    ret, frame = video_capture.read()
                cv2.namedWindow ('student', cv2.WINDOW_AUTOSIZE)
                cv2.resizeWindow('student',800,800);
                frame2 = cv2.resize(frame, (0,0), fx=4, fy=4);
                cv2.imshow('student',frame2)
                cv2.waitKey(0)
                if c <= args.threshold:  # 0.15 is kept as threshold for known face.
                    print ("Unknown student.")
                    cv2.imwrite("student_unknown.jpg", frame)
                    studentData = {'ImagePath':os.path.abspath("student_unknown.jpg")}
                    send_notification(studentData)
                else:
                    print ("Student: {}".format(persons))
                    print ("Identification: {}".format(ss.getInfo(persons, "identification")))
                    print ("Bus_id: {}".format(ss.getInfo(persons, "bus_id")))
                    ss.setInfo(persons, "status", ss.STUDENT_OK_BOARDED_BUS)
                    cv2.imwrite("student.jpg", frame)
                    ss.setInfo(persons, "photo_path", os.path.abspath("student.jpg"))
                    print ("Status: {}".format(ss.getInfo(persons, "status")))
                    print ("Photo Path: {}".format(ss.getInfo(persons, "photo_path")))
                    studentData = {'ID':ss.getInfo(persons, "identification")}
                    send_notification(studentData)
                    # Print the person name and conf value on the frame
            cv2.imshow('', frame)
            for i in range (5):
                video_capture.read()
            # quit the program on the press of key 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()
    else:
       img = cv2.imread(args.image)
       persons, confidences = infer(img, args) 
       for i, c in enumerate(confidences):
           if c <= args.threshold:  # 0.15 is kept as threshold for known face.
               print ("Unknown student.")
               persons[i] = "_unknown"
               studentData = {'ImagePath':os.path.abspath(args.image)}
               send_notification(studentData)
           else:
               print ("Student: {}".format(persons))
               print ("Identification: {}".format(ss.getInfo(persons, "identification")))
               print ("Bus_id: {}".format(ss.getInfo(persons, "bus_id")))
               ss.setInfo(persons, "status", ss.STUDENT_OK_BOARDED_BUS)
               ss.setInfo(persons, "photo_path", args.image)
               print ("Status: {}".format(ss.getInfo(persons, "status")))
               print ("Photo Path: {}".format(ss.getInfo(persons, "photo_path")))
               studentData = {'ID':ss.getInfo(persons, "identification")}
               send_notification(studentData)

