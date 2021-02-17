import cv2
import numpy as np
import re

BLUE = 0
GREEN = 1
RED = 2

# 특정한 색상의 모든 단어가 포함된 이미지 추출
def get_chars(image, color) : #만약, 어떠한 색상이 들어왔다고 가정했을 때
    # 해당 색상이 아닌 다른 두 가지 값을 담아둔다. (RGB 값에서 나머지 값들)
    other_1 = (color + 1) % 3
    other_2 = (color + 2) % 3

    c = image[:, :, other_1] == 255 #만약 해당하는 값이 FF 라면,
    image[c] = [0, 0, 0] # 검정색으로 해당 색깔을 바꾸기
    c = image[:, :, other_2] == 255
    image[c] = [0, 0, 0]
    c = image[:, :, color] < 170 #선택한 값이 AA라면 (두 가지 색상이 섞인 경우)
    image[c] = [0, 0, 0] #검정색으로 바꾸기
    c = image[:, :, color] != 0
    image[c] = [255, 255, 255] #남은 부분들은 하얀색으로 바꾸기

    return image #결론적으로 해당 색의 이미지들만 남게 된다!


#전체 이미지에서 왼쪽부터 단어별로 이미지 추출
def extract_chars(image) :
    chars = []
    colors = [BLUE, GREEN, RED]
    for color in colors:
        image_from_one_color = get_chars(image.copy(), color)  # 색상별로 이미지 추출
        image_gray = cv2.cvtColor(image_from_one_color, cv2.COLOR_BGR2GRAY) #추출된 이미지를 gray로(threshold를 적용하려면 흑백 이미지를 사용)
        ret, thresh = cv2.threshold(image_gray, 127, 255, 0)
        # RETR_EXTERNAL 옵션으로 숫자의 외곽 분리
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #contour 찾기

        for contour in contours :
            #추출된 이미지 크기가 50 이상인 경우에만 실제 문자 데이터라고 파악
            #즉, 작은 픽셀 데이터라면 무시하고 큰 데이터만 문자 데이터라고 판단함.
            area = cv2.contourArea(contour)
            if area > 50 :
                x, y, width, height = cv2.boundingRect(contour) #해당 부분을 사각형 형태의 contour로 추출
                roi = image_gray[y : y+height, x : x+width] #숫자에 해당하는 부분만 뽑기
                chars.append((x, roi))

    chars = sorted(chars, key = lambda char: char[0]) #추출된 이미지를 x축 기준으로 정렬
    #왼쪽부터 차례대로 각각의 이미지를 정려시켜 하나의 수식으로 만든다고 생각하기
    return chars


#이미지를 (20, 20)으로 scaling 해 주기
def resize20(image) :
    resized = cv2.resize(image, (20, 20))
    return resized.reshape(-1, 400).astype(np.float32)


def remove_first_0(string) :
    temp = []
    for i in string :
        if i == '+' or i == '-' or i == '*' :
            temp.append(i)
    # +, 0, * 기호에 대해서 수식을 split 해주기
    split = re.split('\*|\+|-', string)
    i = 0
    temp_count = 0
    result = ""
    for a in split :
        # 왼쪽에 있는 0들을 lstrip으로 지워버리기
        a = a.lstrip('0')
        # 만약 000 이런 경우 다 지워지면 빈 문자열이 되니까 0으로 만들어주기.
        if a == '' :
            a = '0'
        result += a
        if i < len(split) - 1 :
            # 연산자를 다시 붙여주기
            result += temp[temp_count]
            temp_count = temp_count + 1
        i = i + 1
    return result
