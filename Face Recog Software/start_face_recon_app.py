#!/usr/bin/env python
""" Starting script for the face recognition system.
"""

import sys
import os
import numpy as np
from face_recognition_system.videocamera import VideoCamera
from face_recognition_system.detectors import FaceDetector
import face_recognition_system.operations as op
import cv2
from cv2 import __version__
import time



def get_images(frame, faces_coord, shape):
    """ Perfrom transformation on original and face images.

    This function draws the countour around the found face given by faces_coord
    and also cuts the face from the original image. Returns both images.

    :param frame: original image
    :param faces_coord: coordenates of a rectangle around a found face
    :param shape: indication of which shape should be drwan around the face
    :type frame: numpy array
    :type faces_coord: list of touples containing each face information
    :type shape: String
    :return: two images containing the original plus the drawn contour and
             anoter one with only the face.
    :rtype: a tuple of numpy arrays.
    """
    eye_cascade = cv2.CascadeClassifier('face_recognition_system/haarcascade_eye.xml')
    if (shape == "rectangle"):
        faces_img = op.cut_face_rectangle(frame, faces_coord)
        frame = op.draw_face_eye_rectangle(frame, faces_coord, eye_cascade)
    elif (shape == "ellipse"):
        faces_img = op.cut_face_ellipse(frame, faces_coord)
        frame = op.draw_face_ellipse(frame, faces_coord)
    faces_img = op.normalize_intensity(faces_img)
    faces_img = op.resize(faces_img)
    return (frame, faces_img)

def add_person(people_folder, shape, netcam):
    """ Funtion to add pictures of a person
	
    :param people_folder: relative path to save the person's pictures in
    :param shape: Shape to cut the faces on the captured images:
                  "rectangle" or "ellipse"
    :type people_folder: String
    :type shape: String
    """
    person_name = input('What is the name of the new person: ').lower()
    folder = people_folder + person_name
    if not os.path.exists(folder):
        input("Ready to take 20 pictures. Press ENTER when ready.")
        os.mkdir(folder)
        
    video = VideoCamera(netcam)
    detector = FaceDetector('face_recognition_system/frontal_face.xml')
    counter = 1
    timer = 0
    cv2.namedWindow('Video Feed', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Saved Face', cv2.WINDOW_NORMAL)
    while counter < 21:
        frame = video.get_frame()
        face_coord = detector.detect(frame)
        if len(face_coord):
            [frame, face_img] = get_images(frame, face_coord, shape)
            # save a face every second, we start from an offset '5' because
            # the first frame of the camera gets very high intensity
            # readings.
            if timer % 70 == 5: # 1 Second is = 100, less than a sec is < 100
                cv2.imwrite(folder + '/' + str(counter) + '.jpg',
                            face_img[0])
                print ('Images Saved:' + str(counter))
                counter += 1
                cv2.imshow('Saved Face', face_img[0])

        cv2.imshow('Video Feed', frame)
        timer += 5
        k = cv2.waitKey(50) & 0xff
        if k == 27:
            break
        
    del video
    cv2.destroyAllWindows()

def check_choice():
    """ Check if choice is good """
    is_valid = 0
    while not is_valid:
            choice = int(input('Enter your choice [1-3] : '))
            if (choice in [1, 2, 3]):
                is_valid = 1
            else:
                print ("'%d' is not an option.\n" % choice)
    return choice

def recognize_people(people_folder, shape, netcam):
    """ Start recognizing people in a live stream with your webcam

    :param people_folder: relative path to save the person's pictures in
    :param shape: Shape to cut the faces on the captured images:
                  "rectangle" or "ellipse"
    :type people_folder: String
    :type shape: String
    """
    try:
        people = [person for person in os.listdir(people_folder)]
    except:
        print ("Have you added at least one person to the system?")
        sys.exit()
    print ("This are the people in the Recognition System:")
    for i, person in enumerate(people):
        iplus = i+1
        if(iplus < len(people)):
            print (str(i+1) + "- " + person + ",")
            iplus += 1
        elif(iplus == len(people)):
            print (str(i+1) + "- " + person)

    detector = FaceDetector('face_recognition_system/frontal_face.xml')
    
    recognizer = cv2.face.createLBPHFaceRecognizer()
    
    images = []
    labels = []
    labels_people = {}
    for i, person in enumerate(people):
        labels_people[i] = person
        for image in os.listdir(people_folder + person):
            images.append(cv2.imread(people_folder + person + '/' + image, 0))
            labels.append(i)
    try:
        recognizer.train(images, np.array(labels))
    except:
        print ("\nOpenCV Error: Image Dimension Problems or you dont have at least two people in the database\n")
        #sys.exit()
        cv2.waitKey(100)

    video = VideoCamera(netcam)
    k = 1
    threshsum = 0
    data = []
    f= open("predictdata.txt","w+")
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    localtime = time.asctime(time.localtime(time.time()))
    l = ''.join(e for e in localtime if e.isalnum())
    out = cv2.VideoWriter(l + '.avi',fourcc, 20.0, (640,480))
    
    while True:
        frame = video.get_frame()
        faces_coord = detector.detect(frame, True)
        if len(faces_coord):
            [frame, faces_img] = get_images(frame, faces_coord, shape)
            for i, face_img in enumerate(faces_img):
                if (__version__ == "3.1.0"):
                    collector = cv2.face.MinDistancePredictCollector()
                    recognizer.predict(face_img, collector)
                    conf = collector.getDist()
                    pred = collector.getLabel()
                else:
                    [pred, conf] = recognizer.predict(face_img)
                print ("Prediction: " + str(pred))
                print ('Confidence: ' + str(round(conf)))
                avg = 20
                if (k <= avg):
                    threshsum = conf + threshsum
                    threshold = 150
                    k = k + 1
                else:
                    threshold = threshsum/(avg-1)
                print ('Threshold Used: ' + str(threshold))
                print ("k val: " + str(k))
                print ("threshsum: " + str(threshsum))
                if (conf <= threshold and k > avg):
                    cv2.putText(frame, 'Pred => ' + labels_people[pred].capitalize(),
                                (faces_coord[i][0], faces_coord[i][1] - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 1,
                                cv2.LINE_AA)
                    data.append(pred)
                    f.write("%d\n" %(pred))
                elif (k <= avg):
                    cv2.putText(frame, "Finding Threshold",
                                (faces_coord[i][0], faces_coord[i][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1,
                                cv2.LINE_AA)
                elif(conf >= threshold and k > avg):
                    cv2.putText(frame, "Face Unknown",
                                (faces_coord[i][0], faces_coord[i][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2,
                                cv2.LINE_AA)
                    dat = 'NaN'
                    data.append(dat)
                    f.write("%s\n" %(dat))
        print ('Appended Data: ' + str(data))
        cv2.putText(frame, "Press ESC to Exit", (5, frame.shape[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)
        if len(frame):
            # write the flipped frame
            out.write(frame)
            cv2.putText(frame, "Recording...", (490, frame.shape[0] - 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow('Video Feed', frame)
        if cv2.waitKey(100) & 0xFF == 27:
            f.close()
            out.release()
            cv2.destroyAllWindows()
            sys.exit()

## Run the program
if (__name__ == '__main__'):
    print (30 * '-')
    print ("   FACE DETECTION AND RECOGNITION SOFTWARE (Designed by Emmanuel C. Paul, Unilorin 2016/2017 -  13/56EB154")
    wbc = cv2.VideoCapture(0)
    ret, frm = wbc.read()
    if ret == True:
        print (30 * '-')
        print ("Web Camera detected at port 0!")
        print (30 * '-')
        camsource = int(input('Do you want to change camera source? Enter 1 for Yes, 0 for No:> '))
        if (camsource in [1, 2] and camsource == 1):
            netaddress = input('Input your Netcam IP Address in the form, - 10.11.191.2:8080 :> ')
        elif(camsource == 0):
            netaddress = 0
        else:
            print ("'%d' is not an option.\n" % camsource)
    
    print ("Testing Camera....")
    
    if(netaddress == 0):
        NETCAM = 0   
    else:
        NETCAM = "http://" + netaddress + "/video" 
    
    wcam = cv2.VideoCapture(NETCAM)
    ret, frame = wcam.read()
    if ret == True:
        print("Camera is ok. Initialization Complete...!")
    else:
        print("Failed to Initialize Camera for detection process...!")
    wcam.release()
    
    print (30 * '-')
    print ("1. Collect Dataset")
    print ("2. Start Recognizer")
    print ("3. Exit Program")
    print (30 * '-')

    CHOICE = check_choice()

    PEOPLE_FOLDER = "face_recognition_system/people/"
    SHAPE = "rectangle"
    
    
    if CHOICE == 1:
        if not os.path.exists(PEOPLE_FOLDER):
            os.makedirs(PEOPLE_FOLDER)
        add_person(PEOPLE_FOLDER, SHAPE, NETCAM)
    elif CHOICE == 2:
        recognize_people(PEOPLE_FOLDER, SHAPE, NETCAM)
    elif CHOICE == 3:
        sys.exit()
