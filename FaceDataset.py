import cv2
import numpy as np

camera_port=0
cap=cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip=0
offset=20
dataset_path="./Data/"
face_data=[]
filename=input("Enter your first name:")
fullpath=dataset_path+filename+'.npy'
while True:
    return_val,frame=cap.read()
    if return_val==False:
        continue
    gray_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(frame, 1.05, 5)
    faces=sorted(faces, key= lambda f:f[2]*[3])
    for (x, y, w, h) in faces[-1:]:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        frame_section=gray_frame[y-offset: y+h+offset, x-offset: x+w+offset]
        frame_section=cv2.resize(frame_section, (100, 100))
        cv2.imshow("Video Frame", frame)
        cv2.imshow("Video Frame Section", frame_section)
    skip+=1
    if skip==10 and return_val==True:
        print("('_')")
        skip=0
        face_data.append(frame_section)
    key_pressed=cv2.waitKey(1)&0xFF
    if key_pressed==ord('q'):
        break
face_data=np.asarray(face_data)
face_data=face_data.reshape(face_data.shape[0],-1)
print(face_data)
np.save(fullpath, face_data)
print("Successfully saved: {}".format(fullpath))
cap.release()
cv2.destroyAllWindows()