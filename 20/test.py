import cv2
import utils

image = cv2.imread('1.png', cv2.IMREAD_COLOR)
blue = utils.get_chars(image.copy(), utils.BLUE) #파란색 추출
green = utils.get_chars(image.copy(), utils.GREEN) #초록색 추출
red = utils.get_chars(image.copy(), utils.RED) #빨간색 추출

#해당 이미지들 추출
cv2.imshow('Image Gray', blue)
cv2.waitKey(0)
cv2.imshow('Image Gray', green)
cv2.waitKey(0)
cv2.imshow('Image Gray', red)
cv2.waitKey(0)
