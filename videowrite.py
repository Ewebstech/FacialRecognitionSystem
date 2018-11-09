# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 07:32:11 2017

@author: EMMANUEL C. PAUL
"""

import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
localtime = time.asctime(time.localtime(time.time()))
l = ''.join(e for e in localtime if e.isalnum())
out = cv2.VideoWriter(l + '.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()