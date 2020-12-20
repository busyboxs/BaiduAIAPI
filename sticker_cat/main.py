from utils import *
import cv2


image_file = 'faces/01.jpg'
image = cv2.imread(image_file)
image = add_sticker_cat(image)
cv2.imshow('cat', image)
cv2.waitKey()
