import cv2
import paddlehub as hub
import numpy as np
from PIL import Image

hub.logger.setLevel('ERROR')


def blend_images(fore_image, base_image, ratio, pos=None):
    """
    将抠出的人物图像换背景
    fore_image: 前景图片，抠出的人物图片
    base_image: 背景图片
    ratio: 调整前景的比例
    pos: 前景放在背景的位置的，格式为左上角坐标
    """
    if isinstance(base_image, str):
        bg_img = cv2.imread(base_image)  # read background image
    else:
        bg_img = base_image
    if isinstance(fore_image, str):
        fg_img = cv2.imread(fore_image, -1)  # read foreground image
    else:
        fg_img = fore_image
    height_fg, width_fg, _ = fg_img.shape  # get height and width of foreground image
    height_bg, width_bg, _ = bg_img.shape  # get height and width of background image
    if ratio > (height_bg / height_fg):
        print(f'ratio is too large, use maximum ratio {(height_bg / height_fg): .2}')
        ratio = round((height_bg / height_fg), 1)
    if ratio < 0.1:
        print('ratio < 0.1, use minimum ratio 0.1')
        ratio = 0.1
    # if no pos arg input, use this as default
    if not pos:
        pos = (height_bg - int(ratio * height_fg), width_bg // 4)

    roi = bg_img[pos[0]: pos[0] + int(height_fg * ratio), pos[1] : pos[1]+int(width_fg*ratio)]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    if isinstance(fore_image, str):
        fore_image = Image.open(fore_image).resize(roi.shape[1::-1])
    else:
        fore_image = cv2.resize(fore_image, roi.shape[1::-1])

    # 图片加权合成
    scope_map = np.array(fore_image)[:, :, -1] / 255
    scope_map = scope_map[:, :, np.newaxis]
    scope_map = np.repeat(scope_map, repeats=3, axis=2)
    res_image = np.multiply(scope_map, np.array(fore_image)[:, :, 2::-1]) + np.multiply((1 - scope_map), np.array(roi))

    bg_img[pos[0]: pos[0] + roi.shape[0], pos[1]: pos[1] + roi.shape[1]] = np.uint8(res_image)[:, :, ::-1]
    return bg_img


module = hub.Module(name="deeplabv3p_xception65_humanseg")


def seg(image, back_img):
    input_dict = {"image": [image]}
    result, seg_img = module.segmentation(data=input_dict)
    seg_img = seg_img.astype(np.uint8)
    img = blend_images(seg_img, back_img, 1, (0, 0))
    cv2.imshow('seg1', img)


def video():
    capture = cv2.VideoCapture('01.mp4')
    capture.set(1, 250)
    capture2 = cv2.VideoCapture('02.mp4')
    while capture2.isOpened():
        _, frame = capture.read()
        _, background = capture2.read()
        # cv2.imshow('movie', frame)
        seg(frame, background)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    capture.release()
    capture2.release()


video()
