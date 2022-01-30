from PIL import Image
import cv2
import numpy as np

def shadow_crop(image):
    mask_res = image
    img_original = cv2.cvtColor(np.array(mask_res), cv2.COLOR_RGB2BGR)    # Read Pil image to cv2
    # 이미지 외각선 추출 후 크기 맞춰 이미지 자르기
    image_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)    # Convert image to Gray
    blur = cv2.GaussianBlur(image_gray, ksize=(5, 5), sigmaX=0)
    # ret, thresh1 = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    edged = cv2.Canny(blur, 10, 250)   # Draw image with edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))   # Taking all part of additional
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # Finding mask image lines
    contours_xy = np.array(contours)     # Image convert to array

    # Calculate x min max
    x_min, x_max = 0, 0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][0])  # 네번째 괄호가 0일때 x의 값
            x_min = min(value)
            x_max = max(value)

    # y min max
    y_min, y_max = 0, 0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][1])  # 네번째 괄호가 0일때 x의 값
            y_min = min(value)
            y_max = max(value)

    xmin = x_min   # Founding xmin point
    ymin = y_min   # Founding ymin point
    xmax = x_max   # Founding xmax point
    ymax = y_max   # Founding ymax point

    return xmin, ymin, xmax, ymax