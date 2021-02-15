import os
import cv2
import utils

# training_data 폴더를 생성하고, 그 내부에 0~12까지의 폴더를 생성해 준다.
image = cv2.imread("5.png")
chars = utils.extract_chars(image) #색상별로 숫자 이미지 추출

for char in chars :
    #숫자 이미지 출력
    cv2.imshow('Image', char[1])
    input = cv2.waitKey(0) #사용자한테 0~9까지 직접 숫자를 입력을 받음.
    resized = cv2.resize(char[1], (20, 20)) #크기를 (20, 20)으로 바꾸기

    # 사용자가 입력한 숫자가 0~9 사이라면
    if input >= 48 and input <= 57 :
        name = str(input - 48) #입력한 숫자의 폴더에 해당 이미지가 들어가도록.
        #즉, 만약 사용자가 '9'라는 이미지를 보고 9를 입력했다면, 해당 이미지는 '9' 폴더에 들어감
        file_count = len(next(os.walk('./training_data/' + name + '.'))[2])
        cv2.imwrite('./training_data/' + str(input-48) + '/' + str(file_count + 1) + '.png', resized)

    # +, -, *에 대해서는 사용자가 a, b, c를 입력 받아서 분류
    #실제로 폴더에는 10, 11, 12에 들어가게 됨.
    elif input == ord('a') or input == ord('b') or input== ord('c') :
        name = str(input- ord('a') + 10)
        file_count = len(next(os.walk('./training_data/' + name + '.'))[2])
        cv2.imwrite('./training_data/' + name + '/' + str(file_count + 1) + '.png', resized)
