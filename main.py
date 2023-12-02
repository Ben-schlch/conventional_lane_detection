from Calibration import Calibration
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from cv2.typing import MatLike as Mat
from preprocess import preprocess


class LaneDetection:

    def __init__(self):
        self.calibration = Calibration()
        pass

    def run(self):
        video = cv.VideoCapture('./images/Udacity/project_video.mp4')
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame = self.detect_lane(frame)
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()
        cv.destroyAllWindows()

    def detect_lane(self, img: Mat) -> Mat:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = self.calibration.undistort(img)
        img_processed = preprocess(img)
        lines = self._hough_transform(img_processed)
        img = self._draw_lines(img, lines)
        return img

    def _hough_transform(self, img: Mat) -> Mat:
        lines = cv.HoughLinesP(img, rho=2, theta=np.pi / 180, threshold=100, minLineLength=100,
                               maxLineGap=50, lines=np.array([]))
        return lines

    def _seperate_lane_lines(self, lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left_line: list[list[int]] = [[], []]
        right_line: list[list[int]] = [[], []]
        for line in lines:
            for x1, y1, x2, y2 in line:
                gradient = (y2 - y1) / (x2 - x1)
                if gradient < 0:
                    left_line[0].append(x1)
                    left_line[1].append(y1)
                    left_line[0].append(x2)
                    left_line[1].append(y2)
                else:
                    right_line[0].append(x1)
                    right_line[1].append(y1)
                    right_line[0].append(x2)
                    right_line[1].append(y2)
        return np.array(left_line), np.array(right_line)

    def _fit_lane_lines(self, lines: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        left_line, right_line = self._seperate_lane_lines(lines)
        if left_line[0].size == 0 or left_line[1].size == 0:
            left_line = None
        if right_line[0].size == 0 or right_line[1].size == 0:
            right_line = None
        if left_line is not None:
            left_line = np.polyfit(left_line[0], left_line[1], 2)
        if right_line is not None:
            right_line = np.polyfit(right_line[0], right_line[1], 2)
        return left_line, right_line

    def _draw_lines(self, img: Mat, lines: np.ndarray) -> Mat:
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
            y_right = right_line[0] * x**2 + right_line[1] * x + right_line[2]
            x_right = x[y_right > img.shape[0] // 2 + 75]
            y_right = y_right[y_right > img.shape[0] // 2 + 75]
            x_right = x_right[y_right < img.shape[0]]
            y_right = y_right[y_right < img.shape[0]]
            for i in range(len(x_right)):
                cv.circle(line_img, (int(x_right[i]), int(y_right[i])), 5, (255, 0, 0), -1)

        # for line in lines:
        #     for x1, y1, x2, y2 in line:
        #         cv.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)
        img = cv.addWeighted(img, 0.8, line_img, 1, 0)
        return img


if __name__ == '__main__':
    LaneDetection().run()
