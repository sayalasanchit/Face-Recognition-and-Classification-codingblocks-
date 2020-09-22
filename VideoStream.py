import cv2

capture=cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not capture.isOpened():
    raise IOError("Cannot open webcam")
while True:
    ret, frame=capture.read()
    if ret==False:
        continue
    gray_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Video frame", frame)
    cv2.imshow("Gray frame", gray_frame)
    key_pressed=cv2.waitKey(1)&0xff
    if key_pressed==ord('q'):
        break

capture.release()
cv2.destroyAllWindows()