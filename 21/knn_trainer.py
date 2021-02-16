import os
import cv2
import numpy as np

file_names = list(range(0, 13)) #폴더 이름에 해당하는 0~12까지의 리스트
train = []
train_labels =[]

for file_name in file_names :
    path = './training_data/' + str(file_name) + '/' #해당 파일 불러오기
    file_count = len(next(os.walk(path))[2]) #해당 이미지 파일이 몇 개인지 확인
    for i in range(1, file_count + 1) :
        img = cv2.imread(path + str(i) + '.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #train image/label 데이터를 각각 리스트에 넣기
        train.append(gray)
        train_labels.append(file_name)

x = np.array(train)
train = x[:, :].reshape(-1, 400).astype(np.float32) #1차원 배열로 resize
train_labels = np.array(train_labels)[:, np.newaxis] #label 역시 배열 형태로 바꾸어줌

print(train.shape)
print(train_labels.shape)

np.savez("trained.npz", train = train, train_labels = train_labels)
