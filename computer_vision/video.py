import cv2 as cv
cap = cv.VideoCapture(0)
classifier =  cv.CascadeClassifier("haarcascade_eyes.xml")
while True:
        ret,frame =cap.read()
        if ret:
            faces= classifier.detectMultiScale(frame)
            for face in faces:
                x,y,w,h = face
                frame = cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),4)
            cv.imshow("my window",frame)
        key = cv.waitKey(1000)
        if key == ord("q"):
            cv.imwrite("image.jpg",frame)
            break
cap.release()
cv.destroyAllWindows()