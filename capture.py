import cv2

cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('Classifiers/face.xml')
i=0
offset=50

#input unique id to identify at recognition
name=raw_input('enter your id : ')

x,y,w,h = 0,0,0,0
while True:
    ret, im =cam.read()
    #gray scale for capturing
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #capture face from video frame
    faces=detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in faces:
        x,y,w,h = x,y,w,h
        i=i+1
        #saving image in dataset
        cv2.imwrite("dataSet/face-"+name +'.'+ str(i) + ".jpg", gray[y-offset:y+h+offset,x-offset:x+w+offset])
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)        
        cv2.waitKey(10)
    try :
        #show capturing
        cv2.imshow('capture face',im[y-offset:y+h+offset,x-offset:x+w+offset])
    except :
        pass
    if i>10:
        #capture 10 frames and exit
        cam.release()
        cv2.destroyAllWindows()
        break

