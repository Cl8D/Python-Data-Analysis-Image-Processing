import numpy as np
import cv2
import utils

FILE_NAME = "trained.npz" #학습된 데이터 불러오기

# 각 글자의 (1x400) 데이터와 정답 값(0~9, +, *, -)
with np.load(FILE_NAME) as data :
    train = data['train']
    train_labels = data['train_labels']

knn = cv2.ml_KNearest.create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels) #학습시키기

def check(test, train, train_labels) :
    #가장 가까운 k개의 글자를 찾아서, 어떤 숫자에 해당하는지 찾기.
    ret, result, neighbours, dist = knn.findNearest(test, k=1)
    return result

#입력된 이미지를 읽어들여서 하나의 수식으로(result_string)으로 만들기.
#만약 이미지가 123+456이라고 했을 때, 각각을 1, 2, 3..., 5, 6 이런 식으로 읽어들인 다으메 분석하여
#result string 값으로 "123 + 456" 이렇게 만들어준다.

def get_result(file_name) :
    image = cv2.imread(file_name)
    chars = utils.extract_chars(image)
    result_string = ""
    for char in chars :
        matched = check(utils.resize20(char[1]), train, train_labels)
        if matched < 10 :
            result_string += str(int(matched))
            continue
        if matched == 10 :
            matched = '+'
        elif matched == 11 :
            matched = '-'
        elif matched == 12 :
            matched = '*'
        result_string += matched
    return result_string

print(get_result(("1.png")))
