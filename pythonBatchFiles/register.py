import cv2
import sqlite3
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
from torch.utils.data import DataLoader
from torchvision import datasets
import torch


regno = input("Enter your registration number: ")
name = input("Enter your name: ")
email = input("Enter your email: ")
phone = input("Enter your phone number: ")

resnet = InceptionResnetV1(pretrained='vggface2').eval() 
mtcnn = MTCNN() # initializing mtcnn for face detection

def takeImages():
    cam = cv2.VideoCapture(0)

    noOfImages = 0
    while True:
        _, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bounding_boxes, conf, landmarks = mtcnn.detect(img, landmarks=True)
        if bounding_boxes is not None:
            for i in range(len(bounding_boxes)):
                x1, y1, x2, y2 = bounding_boxes[i]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),(0, 0, 255), 2)
                # incrementing sample number
                noOfImages += 1
                # saving the captured face in the dataset folder TrainingImage
                if not os.path.exists(f".\\Dataset\\data\\{int(regno)}"):
                    os.mkdir(f".\\Dataset\\data\\{int(regno)}")
                cv2.imwrite(f".\\Dataset\\data\\{regno}\\{noOfImages}.jpg", gray[y1:y2,x1:x2])
                # display the frame
                cv2.imshow('Taking Images...', img)
            # wait for esc key or q
        if cv2.waitKey(100) and 0xFF == ord('q'):
            break
        # break if the sample number is more than 60. Meaning Images are more than 60.
        elif noOfImages >= 30:
            break
    cam.release()
    cv2.destroyAllWindows()


takeImages()

dataset = datasets.ImageFolder('./Dataset/data') # photos folder path 
idx_to_class = {i:c for c,i in dataset.class_to_idx.items()} # accessing names of peoples from folder names

def collate_fn(x):
    return x[0]

loader = DataLoader(dataset, collate_fn=collate_fn)

name_list = [] # list of names corrospoing to cropped photos
embedding_list = [] # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

for img, idx in loader:
    face, prob = mtcnn(img, return_prob=True)
    if face is not None and prob>0.75:
        emb = resnet(face.unsqueeze(0))
        embedding_list.append(emb.detach())
        name_list.append(idx_to_class[idx])

# save data
data = [embedding_list, name_list] 
torch.save(data, './models/faceRecognition.pt') # saving data.pt file

conn = sqlite3.connect('covid_database.db')
conn.execute(f"INSERT INTO StudentInfo VALUES ('{regno}','{name}' ,'{email}','{phone}')")
conn.commit()
conn.close()