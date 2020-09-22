import numpy as np
import os
import cv2


distance=lambda x0, x1: np.sqrt(np.sum((x0-x1)**2))
def knn(Train, Test, k=7):
    X=Train[: , :-1]
    Y=Train[: , -1]
    m=Y.size
    dist=[]
    for i in range(m):
        d=distance(X[i], Test)
        dist.append([d, Y[i]])
    dist.sort()
    dist=np.array(dist[:k])
    vals=np.unique(dist[:, 1], return_counts=True)
    index=vals[1].argmax()
    pred=vals[0][index]
    return pred


dataset_path='./Data/'
face_data=[]
labels=[]
names={}
class_id=0
for file in os.listdir(dataset_path):
    if file.endswith('.npy'):
        names[class_id]=file[:-4]
        data_item=np.load(dataset_path+file)
        face_data.append(data_item)
        target=class_id*np.ones((data_item.shape[0], ))
        labels.append(target)
        class_id+=1
face_dataset=np.concatenate(face_data, axis=0)
face_labels=np.concatenate(labels, axis=0).reshape((-1,1))
trainset=np.concatenate((face_dataset, face_labels), axis=1)


capture=cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not capture.isOpened():
    raise IOError("Cannot open webcam")
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
offset=20
while True:
    ret, frame=capture.read()
    if ret==False:
        continue
    gray_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    for (x, y, w, h) in faces:  
        face_section=gray_frame[y-offset: y+h+offset, x-offset: x+w+offset]
        face_section=cv2.resize(face_section, (100, 100))
        out=knn(trainset, face_section.flatten())
        pred_name=names[out]
        cv2.putText(frame, pred_name, (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
    cv2.imshow("Video frame", frame)
    key_pressed=cv2.waitKey(1)&0xff
    if key_pressed==ord('q'):
        break
capture.release()
cv2.destroyAllWindows()