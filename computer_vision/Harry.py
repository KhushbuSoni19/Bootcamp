import cv2
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,back = cap.read()#this is simple reading
    if ret:
        #what u r reading is successfull or not
        #what is our camera reading
        cv2.imshow("image",back)
        if cv2.waitKey(5) == ord('q'):#save the image
            cv2.imwrite('khushbu.png',back)
            break
cap.release()
cv2.destroyAllWindows()