from Calibration import Calibration
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from cv2.typing import MatLike as Mat
from preprocess import preprocess
from time import time


class LaneDetection:

    def __init__(self):
        self.calibration = Calibration()
        self.previous_left_line: Mat | None = None
        self.previous_right_line: Mat | None = None
        self.width = 0
        self.height = 0
        pass

    def run(self):
        video = cv.VideoCapture('./images/Udacity/challenge_video.mp4')
        start_time = time()
        frames = 0
        while video.isOpened():
            frames += 1
            ret, frame = video.read()
            if not ret:
                break
            frame = self.detect_lane(frame)
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        end_time = time()
        print(f"Frames: {frames}")
        print(f"Time: {end_time - start_time}")
        print(f"FPS: {frames / (end_time - start_time)}")
        video.release()
        cv.destroyAllWindows()

    def detect_lane(self, img: Mat) -> Mat:

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = self.calibration.undistort(img)
        img_processed = preprocess(img)
        # img_processed = self.calibration.warp_to_birdseye(img_processed)
        # img = self.calibration.warp_to_birdseye(img)
        self.width = img.shape[1]
        # lines = self._hough_transform(img_processed)
        left_line, right_line = self.seperate_lines_on_thresh(img_processed)
        img = self._draw_lines(img, left_line, right_line)
        return img

    def _hough_transform(self, img: Mat) -> Mat:
        lines = cv.HoughLinesP(img, rho=2, theta=np.pi / 180, threshold=100, minLineLength=50,
                               maxLineGap=50, lines=np.array([]))
        return lines

    def _fit_lane_lines(self, right_line, left_line) -> tuple[np.ndarray | None, np.ndarray | None]:
        # right_line, left_line = self._seperate_lane_lines(lines)
        #print(right_line[0])
        if left_line[0].size == 0 or left_line[1].size == 0:
            left_line = None
        if right_line[0].size == 0 or right_line[1].size == 0:
            right_line = None
        if left_line is not None:
            # print(f"left_line: {left_line}")
            if len(left_line[0]) < 3:
                left_line = self.previous_left_line \
                    if self.previous_left_line is not None \
                    else np.polyfit(left_line[1], left_line[0], 1)

            else:
                left_line = np.polyfit(left_line[1], left_line[0], 2)
                print(f"left_line: {left_line}")
                self.previous_left_line = left_line
        if right_line is not None:
            if len(right_line[0]) < 3:
                print("BBBBBb")
                right_line = self.previous_right_line \
                    if self.previous_right_line is not None \
                    else np.polyfit(right_line[1], right_line[0], 1)
            else:
                right_line = np.polyfit(right_line[1], right_line[0], 2)
                self.previous_right_line = right_line
        return left_line, right_line

    def _draw_lines(self, img: Mat, left_line, right_line) -> Mat:
        # if lines is None:
        #     return img
        left_line, right_line = self._fit_lane_lines(left_line, right_line)
        img = np.copy(img)
        half_height = img.shape[0] // 2
        x = np.linspace(half_height + 85, img.shape[0] -1, img.shape[0] - half_height)
        if len(left_line) == 2:
            y_left = left_line[0] * x + left_line[1]
        elif len(left_line) == 3:
            y_left = left_line[0] * x ** 2 + left_line[1] * x + left_line[2]
        if len(right_line) == 2:
            y_right = right_line[0] * x + right_line[1]
        elif len(right_line) == 3:
            y_right = right_line[0] * x ** 2 + right_line[1] * x + right_line[2]
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        left_points = np.array(np.transpose(np.vstack([y_left, x])), np.int32)
        right_points = np.array(np.transpose(np.vstack([y_right, x])), np.int32)
        cv.polylines(line_img, [left_points], False, (255, 0, 0), thickness=10)
        cv.polylines(line_img, [right_points], False, (0, 0, 255), thickness=10)
        return cv.addWeighted(img, 0.8, line_img, 1, 0)

    def seperate_lines_on_thresh(self, img: Mat):
        middle = self.width // 2
        right_y, right_x = np.where(img[:, middle:] == 255)
        right_y, right_x = right_y.copy(), right_x.copy()
        left_y, left_x = np.where(img[:, :middle] == 255)
        left_y, left_x = left_y.copy(), left_x.copy()

        # print(left_x[np.argmax(left_x)], right_x[np.argmax(right_x)])
        right_x += middle
        # right_y -= middle
        return (left_x, left_y), (right_x, right_y)

    def _debug_draw_lines(self, img: Mat, left_line, right_line):
        """
        Debugmethode um die gefundenen Linien auf die Hough-Transformierten Linien zu zeichnen
        :param img:
        :param lines:
        :return:
        """
        return img


if __name__ == '__main__':
    LaneDetection().run()
