from math import degrees, atan2
import cv2
import paddlehub as hub
import numpy as np


def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()
    # convert 3 channels to 4 channels
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)

    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape
    roi = bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] = cv2.add(img1_bg, img2_fg)

    # convert 4 channels to 3 channels
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img


def angle_between(p1, p2):
    x_diff = p2[0] - p1[0]
    y_diff = p2[1] - p1[1]
    return degrees(atan2(y_diff, x_diff))


def wear_glasses(image, glasses, eye_left_center, eye_right_center):
    eye_left_center = np.array(eye_left_center)
    eye_right_center = np.array(eye_right_center)
    glasses_center = np.mean([eye_left_center, eye_right_center], axis=0)  # put glasses's center to this center
    glasses_size = np.linalg.norm(eye_left_center - eye_right_center) * 2  # the width of glasses mask
    angle = -angle_between(eye_left_center, eye_right_center)

    glasses_h, glasses_w = glasses.shape[:2]
    glasses_c = (glasses_w / 2, glasses_h / 2)
    M = cv2.getRotationMatrix2D(glasses_c, angle, 1)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((glasses_h * sin) + (glasses_w * cos))
    nH = int((glasses_h * cos) + (glasses_w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - glasses_c[0]
    M[1, 2] += (nH / 2) - glasses_c[1]

    rotated_glasses = cv2.warpAffine(glasses, M, (nW, nH))

    try:
        image = overlay_transparent(image, rotated_glasses, glasses_center[0], glasses_center[1],
                                    overlay_size=(
                                        int(glasses_size),
                                        int(rotated_glasses.shape[0] * glasses_size / rotated_glasses.shape[1]))
                                    )
    except:
        print('failed overlay image')
    return image


def get_eye_center_point(landmarks, idx1, idx2):
    center_x = (landmarks[idx1][0] + landmarks[idx2][0]) // 2
    center_y = (landmarks[idx1][1] + landmarks[idx2][1]) // 2
    return center_x, center_y


def main():
    module = hub.Module(name="face_landmark_localization")
    image_file = 'faces/01.jpg'
    glasses_file = './glasses/glasses4.png'

    image = cv2.imread(image_file)
    glasses = cv2.imread(glasses_file, cv2.IMREAD_UNCHANGED)
    result = module.keypoint_detection(images=[image])
    landmarks = result[0]['data'][0]
    eye_left_point = get_eye_center_point(landmarks, 36, 39)
    eye_right_point = get_eye_center_point(landmarks, 42, 45)
    image = wear_glasses(image, glasses, eye_left_point, eye_right_point)
    cv2.imshow('result', image)
    cv2.waitKey()


if __name__ == "__main__":
    main()
