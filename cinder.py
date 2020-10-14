from rimo_utils import cv0
import cv2

import numpy as np


生肉色相 = 6
熟肉色相 = 15


def f(img):
    img = img.copy()
    ori_img = img.copy()
    x, y = img.shape[:2]

    img_hsv = cv0.cvtColor(img, cv0.COLOR_BGR2HSV)

    h = img_hsv[:, :, 0]
    s = img_hsv[:, :, 1]
    v = img_hsv[:, :, 2]

    # 这里是180的范围，但是下面是360
    gh = (h > 165) + (h < 15)
    gh[np.where(s < 60)] = 0
    gh[np.where(v < 40)] = 0
    gh = gh.astype(np.uint8)

    gh = cv0.dilate(gh, np.ones([5, 5]))
    gh = cv0.erode(gh, np.ones([5, 5]))

    contours, _ = cv2.findContours(gh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    good_contours = [c for c in contours if cv2.contourArea(c) > 1000]

    for contour in good_contours:
        mask = np.zeros([x, y], dtype=np.uint8)
        mask = cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

        q = ori_img[np.where(mask == 255)]
        mean = ((q.mean(axis=0).reshape([1, 1, 3]))/255).astype(np.float32)

        mean_hsv = cv0.cvtColor(mean, cv0.COLOR_BGR2HSV)
        肉色相 = mean_hsv[0, 0, 0]
        if 肉色相 > 180:
            肉色相 -= 360

        熟度 = (肉色相 - 生肉色相) / (熟肉色相 - 生肉色相)
        熟度 = max(0, 熟度)
        熟度 = min(1, 熟度)

        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        cv2.drawContours(img, [contour], -1, (0, 255, 255), thickness=3, lineType=16)

        cv0.putText(img, '%d%%' % (熟度*100), (cx+1, cy+1), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 4)
        cv0.putText(img, '%d%%' % (熟度*100), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 3, (255, 128, 128), 4)

    cv0.imshow('125', img)
    cv0.waitKey(1)


s = None
for img in cv0.VideoCapGen(0, [1280, 720]):
    img = img[:, ::-1]
    if s is None:
        s = img
    else:
        s = (s.astype(np.float)*0.7 + img.astype(np.float)*0.3).astype(np.uint8)
    f(s)
