import paddlehub as hub
from random import randrange
import math
import numpy as np
import cv2


def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH)), M


def get_random_color():
    return randrange(0, 255, 1), randrange(10, 255, 1), randrange(10, 255, 1)


LABELS = ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge',
          'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip']
COLORS = [get_random_color() for _ in LABELS]


def get_landmarks(img):
    module = hub.Module(name="face_landmark_localization")
    result = module.keypoint_detection(images=[img])
    landmarks = result[0]['data'][0]
    return landmarks


def get_face_rectangle(img):
    face_detector = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_320")
    result = face_detector.face_detection(images=[img])
    x1 = int(result[0]['data'][0]['left'])
    y1 = int(result[0]['data'][0]['top'])
    x2 = int(result[0]['data'][0]['right'])
    y2 = int(result[0]['data'][0]['bottom'])
    return x1, y1, x2 - x1, y2 - y1


def face_landmarks(face_image, location_of_face=None):
    landmarks = get_landmarks(face_image)
    landmarks_as_tuples = [[(int(p[0]), int(p[1])) for p in landmarks]]
    return [{
        "chin": points[0:17],
        "left_eyebrow": points[17:22],
        "right_eyebrow": points[22:27],
        "nose_bridge": points[27:31],
        "nose_tip": points[31:36],
        "left_eye": points[36:42],
        "right_eye": points[42:48],
        "top_lip": points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],
        "bottom_lip": points[54:60] + [points[48]] + [points[60]] +
                      [points[67]] + [points[66]] + [points[65]] + [points[64]]
    } for points in landmarks_as_tuples]


def get_bound_box(points):
    points = np.array(points)
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    return min_x, min_y, width, height


def face_part(points, part):
    assert part in LABELS, "face_part should be in [" + ','.join(LABELS) + ']'
    x, y, w, h = get_bound_box(points[part])
    return x, y, w, h


def calculate_angle(point1, point2):
    x1, x2, y1, y2 = point1[0], point2[0], point1[1], point2[1]
    return 180 / math.pi * math.atan((float(y2 - y1)) / (x2 - x1))


def add_sticker_cat(img):
    sticker_img = 'stickers/cat.png'
    sticker = cv2.imread(sticker_img, -1)
    landmarks = face_landmarks(img)
    angle = calculate_angle(landmarks[0]['left_eyebrow'][0], landmarks[0]['right_eyebrow'][-1])
    nose_tip_center = [180, 400]  # nose center of sticker
    rotated, M = rotate_bound(sticker, angle)
    tip_center_rotate = np.dot(M, np.array([[nose_tip_center[0]], [nose_tip_center[1]], [1]]))
    sticker_h, sticker_w,  _ = rotated.shape
    x, y, w, h = get_face_rectangle(img)
    dv = w / sticker_w
    distance_x, distance_y = int(tip_center_rotate[0] * dv), int(tip_center_rotate[1] * dv)
    rotated = cv2.resize(rotated, (0, 0), fx=dv, fy=dv)
    sticker_h, sticker_w, _ = rotated.shape
    y_top_left = landmarks[0]['nose_tip'][2][1] - distance_y
    x_top_left = landmarks[0]['nose_tip'][2][0] - distance_x
    for chanel in range(3):
        img[y_top_left:y_top_left + sticker_h, x_top_left:x_top_left + sticker_w, chanel] = \
            rotated[:, :, chanel] * (rotated[:, :, 3] / 255.0) + \
            img[y_top_left:y_top_left + sticker_h, x_top_left:x_top_left + sticker_w, chanel] \
            * (1.0 - rotated[:, :, 3] / 255.0)

    return img