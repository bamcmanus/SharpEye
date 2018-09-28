import numpy as np
import cv2 as cv

face_cascade = cv.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)

while(True):
    #capture frame by frame
    ret,frame = cap.read()

    #convert fram to grayscale
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    #detect face from gray scale frame
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)

    #draw bounding box
    color = (0,0,255) # BGR scale
    stroke = 2 # line width
    for(x,y,w,h) in faces:
        region_of_interest_gray = gray[y:y+h, x:x+w]
        end_x = x+w
        end_y = y+h
        cv.rectangle(frame, (x,y), (end_x,end_y), color, stroke)

    #display the resulting frame
    cv.imshow('frame',frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break

#when complete
cap.release()
cv.destroyAllWindows()
