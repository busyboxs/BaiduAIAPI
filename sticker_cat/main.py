from utils import face_landmarks, get_random_color, COLORS, face_part, calculate_angle, get_face_rectangle
from utils import add_sticker_cat
import cv2
import paddlehub as hub
from imutils import rotate_bound
import math
import numpy as np


image_file = 'faces/01.jpg'
image = cv2.imread(image_file)
image = add_sticker_cat(image)
cv2.imshow('cat', image)
cv2.waitKey()
