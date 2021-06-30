
# face + eye + nose + mouth detection in Video
import cv2,time
import numpy as np

face_cascade = cv2.CascadeClassifier("classifiers\\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("classifiers\\haarcascade_mcs_eyepair_small.xml")
nose_cascade = cv2.CascadeClassifier("classifiers\\haarcascade_mcs_nose.xml")
smile_cascade = cv2.CascadeClassifier("classifiers\\haarcascade_mcs_mouth.xml")

video = cv2.VideoCapture("test_video.mp4")
n = 0
while (video.isOpened()):
    ret,frame = video.read()
    gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img,scaleFactor=1.3, minNeighbors=3)
    for x,y,w,h in faces :
        image = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        image = cv2.putText(frame,"Face",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.50,(0,255,0),2)

        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray,1.3,3)

        for ex,ey,ew,eh in eyes :
            image = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
            image = cv2.putText(roi_color,"Eye",(ex,ey),cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,0,0),2)


        # For Nose

        nose =  nose_cascade.detectMultiScale(roi_gray,1.3,3)

        for nx,ny,nw,nh in nose :
            image = cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,0,255),2)
            image = cv2.putText(roi_color,"Nose",(nx,ny),cv2.FONT_HERSHEY_SIMPLEX,0.50,(0,0,255),2)

        # For Mouth

        smile =  smile_cascade.detectMultiScale(roi_gray,1.9,25)

        for sx,sy,sw,sh in smile :
            image = cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(255,255,0),2)
            image = cv2.putText(roi_color,"Mouth",(sx,sy),cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,255,0),2)


    img_resize = cv2.resize(frame,(int(frame.shape[1]/2),int(frame.shape[0]/2)))
    cv2.imshow("Frame",img_resize)

    key = cv2.waitKey(1)

    if key is ord('q'):
        break

video.release()
cv2.destroyAllWindows()
