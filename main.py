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
        pass

    def run(self):
        video = cv.VideoCapture('./images/Udacity/project_video.mp4')
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
        img_processed = self.calibration.warp_to_birdseye(img_processed)
        img = self.calibration.warp_to_birdseye(img)
        lines = self._hough_transform(img_processed)
        img = self._draw_lines(img, lines)
        return img

    def _hough_transform(self, img: Mat) -> Mat:
        lines = cv.HoughLinesP(img, rho=2, theta=np.pi / 180, threshold=100, minLineLength=50,
                               maxLineGap=50, lines=np.array([]))
        return lines

    def _seperate_lane_lines(self, lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left_line: list[list[int]] = [[], []]
        right_line: list[list[int]] = [[], []]
        middle = 640
        for line in lines:
            for x1, y1, x2, y2 in line:
                # gradient = (y2 - y1) / (x2 - x1)
                if x1 < middle and x2 < middle:
                    left_line[0].append(x1)
                    left_line[1].append(y1)
                    left_line[0].append(x2)
                    left_line[1].append(y2)
                elif x1 > middle and x2 > middle:
                    right_line[0].append(x1)
                    right_line[1].append(y1)
                    right_line[0].append(x2)
                    right_line[1].append(y2)
        # left_line[0].append(200)
        # left_line[1].append(628)
        # right_line[0].append(1100)
        # right_line[1].append(628)
        # print(right_line[0][np.argmin(right_line[0])])
        # print(right_line[1])
        # left_line_array = np.array(np.array(left_line[0]), np.array(left_line[1]))
        left_line_array = np.array(left_line)
        right_line_array = np.array(right_line)
        left_line_array[0].sort()
        right_line_array[0].sort()
        left_line_array[1].sort()
        right_line_array[1].sort()
        # print(right_line_array[1])
        # print(left_line_array[0])
        return right_line_array, left_line_array

    def _fit_lane_lines(self, lines: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        right_line, left_line = self._seperate_lane_lines(lines)
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
                    else np.polyfit(left_line[0], left_line[1], 1)
                # print(f"left_line: {left_line[0]}x + {left_line[1]}")
            else:
                original_left_line = np.copy(left_line)
                left_line = np.polyfit(left_line[0], left_line[1], 2)
                #
                # if abs(left_line[0]) > 2 or abs(left_line[1]) > 5 or abs(left_line[2]) > 200:
                #     left_line = np.polyfit(original_left_line[0], original_left_line[1], 1)
                #     print(f"left_line: {left_line[0]}x + {left_line[1]}")
                # else:
                #     print(f"left_line: {left_line[0]}x^2 + {left_line[1]}x + {left_line[2]}")
                self.previous_left_line = left_line
        if right_line is not None:
            # print(f"right_line: {right_line}")
            if len(right_line[0]) < 3:
                right_line = self.previous_right_line \
                    if self.previous_right_line is not None \
                    else np.polyfit(right_line[0], right_line[1], 1)
            else:
                original_right_line = np.copy(right_line)
                right_line = np.polyfit(right_line[0], right_line[1], 2)
                #
                # if abs(right_line[0]) > 2 or abs(right_line[1]) > 5 or abs(right_line[2]) > 200:
                #     right_line = np.polyfit(original_right_line[0], original_right_line[1], 1)
                #     print(f"right_line: {right_line[0]}x + {right_line[1]}")
                # else:
                #     print(f"right_line: {right_line[0]}x^2 + {right_line[1]}x + {right_line[2]}")
                self.previous_right_line = right_line
        return left_line, right_line

    def _draw_lines(self, img: Mat, lines: np.ndarray) -> Mat:
        if lines is None:
            return img
        left_line, right_line = self._fit_lane_lines(lines)
        img = np.copy(img)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        x = np.linspace(0, 1280, 1000)
        if left_line is not None:
            if len(left_line) == 2:
                y_left = left_line[0] * x + left_line[1]
            elif len(left_line) == 3:
                y_left = left_line[0] * x**2 + left_line[1] * x + left_line[2]
            x_left = x[y_left > 0]
            y_left = y_left[y_left > 0]
            x_left = x_left[y_left < img.shape[0]]
            y_left = y_left[y_left < img.shape[0]]
            y_left = y_left[x_left < img.shape[1] // 2 + img.shape[1] // 20]
            x_left = x_left[x_left < img.shape[1] // 2 + img.shape[1] // 20]
            for i in range(len(x_left)):
                cv.circle(line_img, (int(x_left[i]), int(y_left[i])), 5, (255, 0, 0), -1)

        if right_line is not None:
            if len(right_line) == 2:
                y_right = right_line[0] * x + right_line[1]
            elif len(right_line) == 3:
                y_right = right_line[0] * x**2 + right_line[1] * x + right_line[2]
            x_right = x[y_right > 0]
            y_right = y_right[y_right > 0]
            x_right = x_right[y_right < img.shape[0]]
            y_right = y_right[y_right < img.shape[0]]
            for i in range(len(x_right)):
                cv.circle(line_img, (int(x_right[i]), int(y_right[i])), 5, (255, 0, 0), -1)

        # for line in lines:
        #     for x1, y1, x2, y2 in line:
        #         cv.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)
        img = cv.addWeighted(img, 0.8, line_img, 1, 0)
        return img

    def _debug_draw_lines(self, img: Mat, lines: np.ndarray):
        """
        Debugmethode um die gefundenen Linien auf die Hough-Transformierten Linien zu zeichnen
        :param img:
        :param lines:
        :return:
        """
        if lines is None:
            return img
        left_line, right_line = self._fit_lane_lines(lines)
        img = np.copy(img)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        x = np.linspace(0, 1280, 1000)
        if left_line is not None:
            y_left = left_line[0] * x**2 + left_line[1] * x + left_line[2]
            x_left = x[y_left > img.shape[0] // 2 + 75]
            y_left = y_left[y_left > img.shape[0] // 2 + 75]
            x_left = x_left[y_left < img.shape[0]]
            y_left = y_left[y_left < img.shape[0]]
            y_left = y_left[x_left < img.shape[1] // 2 + img.shape[1] // 20]
            x_left = x_left[x_left < img.shape[1] // 2 + img.shape[1] // 20]
            for i in range(len(x_left)):
                cv.circle(line_img, (int(x_left[i]), int(y_left[i])), 5, (255, 0, 0), -1)

        if right_line is not None:
            if len(right_line) == 2:
                y_right = right_line[0] * x + right_line[1]
            elif len(right_line) == 3:
                y_right = right_line[0] * x**2 + right_line[1] * x + right_line[2]
            x_right = x[y_right > img.shape[0] // 2 + 75]
            y_right = y_right[y_right > img.shape[0] // 2 + 75]
            x_right = x_right[y_right < img.shape[0]]
            y_right = y_right[y_right < img.shape[0]]
            for i in range(len(x_right)):
                cv.circle(line_img, (int(x_right[i]), int(y_right[i])), 5, (255, 0, 0), -1)

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)
        # img = cv.addWeighted(img, 0.8, line_img, 1, 0)
        return line_img


if __name__ == '__main__':
    LaneDetection().run()
