import cv2

capture=cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not capture.isOpened():
    raise IOError("Cannot open webcam")
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
while True:
    ret, frame=capture.read()
    if ret==False:
        continue
    gray_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1) #white rectangle
    cv2.imshow("Video frame", frame)
    key_pressed=cv2.waitKey(1)&0xff
    if key_pressed==ord('q'):
        break

capture.release()
cv2.destroyAllWindows()