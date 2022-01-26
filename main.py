import os
from pdb import main

import cv2
import numpy as np
import math

ROI_DICT = {
    "top_left": {
        "h": 200,
        "w": 200,
        "x": 50,
        "y": 150
    },
    "bottom_left": {
        "h": 150,
        "w": 200,
        "x": 50,
        "y": 300,
        "y_shift": 75
    },
    "top_right": {
        "h": 200,
        "w": 200,
        "x": 50,
        "y": 150
    },
    "bottom_right": {
        "h": 200,
        "w": 200,
        "x": 50,
        "y": 150}
}

IMG_DIR = 'C:/Users/F1337/Downloads/Telegram Desktop/naycha_crop/naycha_crop/'


def crop_ROI(img, position):
    x, y, w, h = ROI_DICT[position]["x"], ROI_DICT[position]["y"], ROI_DICT[position]["w"], ROI_DICT[position]["h"]
    return img[y:y + h, x:x + w]


def find_lines(img):
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, 11, -1)
    edges = cv2.Canny(th2, 10, 40, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 360, 60, None, 0, 0)
    horizontal = [None, -100000]
    vertical = [None, 100000, 10000]

    if lines is not None:
        for j in range(0, len(lines)):
            rho = lines[j][0][0]

            theta = lines[j][0][1]
            if abs(theta * 180 / 3.14 - 180) < 20 or abs(theta * 180 / 3.14) < 20:

                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                if x0 < vertical[2]:
                    vertical = [theta, rho, x0]

            if 20 > abs(theta * 180 / 3.14) - 90 > -20:
                if rho > horizontal[1]:
                    horizontal = [theta, rho]
        pv1, pv2, ph1, ph2 = None, None, None, None
        if horizontal[0]:
            a = math.cos(horizontal[0])
            b = math.sin(horizontal[0])
            x0 = a * horizontal[1]
            y0 = b * horizontal[1]
            pth1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pth2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            ph1, ph2 = pth1, pth2
        if vertical[0]:
            a = math.cos(vertical[0])
            b = math.sin(vertical[0])
            x0 = a * vertical[1]
            y0 = b * vertical[1]
            ptv1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            ptv2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            pv1, pv2 = ptv1, ptv2

        return (ph1, ph2), (pv1, pv2)


def line_intersection(line1_1, line2_2):
    xdiff = (line1_1[0][0] - line1_1[1][0], line2_2[0][0] - line2_2[1][0])
    ydiff = (line1_1[0][1] - line1_1[1][1], line2_2[0][1] - line2_2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1_1), det(*line2_2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


if __name__ == "__main__":
    for file in os.listdir(IMG_DIR):
        org_img = cv2.imread(os.path.join(IMG_DIR, file))
        crop_img_top_left = crop_ROI(org_img, "top_left")
        crop_img_top_left = cv2.cvtColor(crop_img_top_left, cv2.COLOR_RGB2GRAY)
        line1_top_left, line2_top_left = find_lines(crop_img_top_left)
        if line1_top_left is not None:
            x_cropped_top_left, y_cropped_top_left = line_intersection(line1_top_left, line2_top_left)
            x_cropped_top_left, y_cropped_top_left = int(x_cropped_top_left), int(y_cropped_top_left)
            crop_img_bottom_left = crop_ROI(org_img, "bottom_left")
            crop_img_bottom_left = cv2.cvtColor(crop_img_bottom_left, cv2.COLOR_RGB2GRAY)
            _, line2_bottom_left = find_lines(crop_img_bottom_left)
            x1, y1, x2, y2 = line1_top_left[0][0], line1_top_left[0][1], line1_top_left[1][0], line1_top_left[1][1]
            line1_bottom_left = (x1, y1 + ROI_DICT["bottom_left"]["y_shift"]), (x2, y2 + ROI_DICT["bottom_left"]["y_shift"])
            print(line1_bottom_left, line2_bottom_left)
            x_cropped_bottom_left, y_cropped_bottom_left = line_intersection(line1_bottom_left, line2_bottom_left)
            x_cropped_bottom_left, y_cropped_bottom_left = int(x_cropped_bottom_left), int(y_cropped_bottom_left)
            x_original_top_left, y_original_top_left = x_cropped_top_left + ROI_DICT["top_left"][
                "x"], y_cropped_top_left + ROI_DICT["top_left"]["y"]
            x_original_bottom_left, y_original_bottom_left = x_cropped_bottom_left + ROI_DICT["bottom_left"][
                "x"], y_cropped_bottom_left + ROI_DICT["bottom_left"]["y"]


        # image = cv2.circle(crop_img_top_left, (x_cropped, y_cropped), radius=3, color=(255, 255, 0), thickness=-1)
            org_img = cv2.circle(org_img, (x_original_top_left, y_original_top_left), radius=3, color=(255, 255, 0), thickness=-1)
            org_img = cv2.circle(org_img, (x_original_bottom_left, y_original_bottom_left), radius=3, color=(255, 0, 0), thickness=-1)
        # cv2.imshow('a', image)
        cv2.imshow('ab', org_img)
        cv2.waitKey(0)
