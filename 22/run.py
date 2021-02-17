import numpy as np
import cv2
import utils
import requests
import shutil
import time

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

host = "http://localhost:10000"
url = '/start'

#target_images라는 폴더 생성
with requests.Session() as s :
    answer = ''
    for i in range(0, 30) : #30개의 이미지를 풀어야 한다.
        start_time = time.time() #수식을 푼 다음에 다시 서버로 전송하는 시간 계산하기
        params = {'ans': answer} #정답값을 넣어서 서버로 전송할 예정

        #정답을 파라미터에 달아서 전송하여, 이미지 경로 받아오기
        response = s.post(host + url, params)
        print('Server Return: ' + response.text)
        if i == 0 :
            returned = response.text #가장 처음에는 이미지 url 정보가 나옴
            image_url = host + returned #url로부터 이미지를 다운받기
            url = '/check'
        else :
            returned = response.json() #json 형태로 데이터가 들어온다
            image_url = host + returned['url'] #image url 정보를 갱신해줌. (계속 새로운 이미지를 받아오니까)
        print('Problem ' + str(i) + ': ' + image_url)

        #특정한 폴더에 이미지 파일을 다운로드 받는다
        response = s.get(image_url, stream = True) #session.get() 함수 이용
        target_image = './target_images/' + str(i) + '.png'
        with open(target_image, 'wb') as out_file :
            shutil.copyfileobj(response.raw, out_file)
        del response

        #다운로드 받은 이미지 파일을 분석하여 답을 도출한다
        answer_string = get_result(target_image) #이미지로부터 하나의 string 값으로 수식 도출
        print('String: ' + answer_string)
        answer_string = utils.remove_first_0(answer_string) #앞쪽에 있는 0들 제거
        answer = str(eval(answer_string)) #eval을 이용해서 해당 문자열 수식 계산하기
        print('Answer: ' + answer) #결과 추출
        print("--- %s seconds ---" % (time.time() - start_time)) #시간이 얼마나 걸린지 계산