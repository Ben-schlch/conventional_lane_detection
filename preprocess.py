import cv2 as cv
import numpy as np
from cv2.typing import MatLike as Mat
from Calibration import Calibration
import matplotlib.pyplot as plt


def preprocess(img: Mat) -> Mat:
    # img = img[img.shape[0] // 2:, :]
    img_filtered = __filter_colors(img)
    img_gray = cv.cvtColor(img_filtered, cv.COLOR_RGB2GRAY)
    img_blurred = cv.GaussianBlur(img_gray, (5, 5), 0)
    img_canny = cv.Canny(img_blurred, 50, 150) # cv.threshold(img_gray, 200, 255, cv.THRESH_BINARY)# cv.Canny(img_gray, 50, 150)
    img_roi = __find_and_cut_region(img_canny)
    img_warped = Calibration().warp_to_birdseye(img_roi)
    # img_warped = cv.morphologyEx(img_warped, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return img_warped


def __filter_colors(img: Mat) -> Mat:
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 75, 50])
    upper_yellow = np.array([40, 255, 255])
    mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv.inRange(img, np.array([200, 200, 200]), np.array([255, 255, 255]))
    mask_yw = cv.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv.bitwise_and(img, img, mask=mask_yw)
    return mask_yw_image


def __find_and_cut_region(img: Mat) -> Mat:
    middle_x = img.shape[1] / 2
    middle_y = 0 # img.shape[0] / 2
    triangle_top = (middle_x, middle_y + 50)
    top_left = (middle_x - 140, middle_y + 100)
    top_right = (middle_x + 140, middle_y + 100)
    triangle_left = (img.shape[1] / 20, img.shape[0])
    triangle_right = (img.shape[1] * 95 / 100, img.shape[0])
    vertices = np.array([[triangle_left, top_left, top_right, triangle_right]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv.fillPoly(mask, vertices, (255, 255, 255))
    masked_image = cv.bitwise_and(img, mask)
    mid_roi = np.float32([[640, img.shape[0] - 220], [350, img.shape[0]], [850, img.shape[0]], [662, img.shape[0] - 220]])
    # mid_roi = np.float32([[550, img.shape[0] - 140], [350, img.shape[0]], [850, img.shape[0]], [700, img.shape[0] - 140]])
    cv.fillPoly(masked_image, np.int32([mid_roi]), (0, 0, 0))
    return masked_image

