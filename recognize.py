import cv2,os
import numpy as np
from PIL import Image 
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "Classifiers/face.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
path = 'dataSet'

cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX #Creates a font
i=0
x,y,h= 20,20,0
while True:
    i+=1
    ret, im =cam.read()
    #gray scale for capturing
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #detect face from video frame
    faces=faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        x,y,h = x,y,h
        #get predicted ID
        nbr_predicted, conf = recognizer.predict(gray[y:y+h,x:x+w])       
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        
    im = cv2.flip(im,1)                
    try:
        cv2.cv2.putText(im,"id = "+str(nbr_predicted), (x,y+h),font,0.8,(255,0,0), 2)
        del nbr_predicted
    except NameError:
        cv2.cv2.putText(im,"unidentified", (x,y+h),font,0.8,(0,0,255), 2)
    
        
    cv2.imshow('recognize face',im)
        
    if cv2.waitKey(1) == 27:
        cam.release()
        cv2.destroyAllWindows()
        break   

