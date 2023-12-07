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
        self.previous_left_line = None
        self.previous_right_line = None
        self.previous_real_left = None
        self.previous_real_right = None
        self.width = 0
        self.height = 0
        self.y_m_per_pix = 30 / 720
        self.x_m_per_pix = 3.7 / 700
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
        # img_warped = self.calibration.warp_to_birdseye(img)
        img_processed = preprocess(img)
        # img_processed = self.calibration.warp_to_birdseye(img_processed)

        self.width = img.shape[1]
        # lines = self._hough_transform(img_processed)
        left_line, right_line = self.seperate_lines_on_thresh(img_processed)
        fit_left_line, fit_right_line, real_left, real_right = self._fit_lane_lines(right_line, left_line)
        radius = self._calculate_curvature(real_left, real_right)
        print(radius)
        img = self._draw_lines(img, fit_left_line, fit_right_line)
        return img

    def _hough_transform(self, img: Mat) -> Mat:
        lines = cv.HoughLinesP(img, rho=2, theta=np.pi / 180, threshold=100, minLineLength=50,
                               maxLineGap=50, lines=np.array([]))
        return lines

    def _fit_lane_lines(self, right_line, left_line) \
            -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        # right_line, left_line = self._seperate_lane_lines(lines)
        # print(right_line[0])
        left_line_coeffs, real_left_coeffs = self.__fit_line(left_line)
        right_line_coeffs, real_right_coeffs = self.__fit_line(right_line)
        if left_line_coeffs is not None:
            self.previous_left_line = left_line_coeffs
            self.previous_real_left = real_left_coeffs
        else:
            left_line_coeffs = self.previous_left_line
            real_left_coeffs = self.previous_real_left
        if right_line_coeffs is not None:
            self.previous_real_right = real_right_coeffs
            self.previous_right_line = right_line_coeffs
        else:
            right_line_coeffs = self.previous_right_line
            real_right_coeffs = self.previous_real_right
        if left_line_coeffs is not None and real_left_coeffs is None:
            print("????")
        return left_line_coeffs, right_line_coeffs, real_left_coeffs, real_right_coeffs

    def __fit_line(self, line: np.ndarray | None):
        if line is None:
            return None, None
        if 0 in line.shape[:2]:
            return None, None
        if len(line[0]) < 3:
            return None
        coeffs = np.polyfit(line[1], line[0], 2)
        if abs(coeffs[0]) > 0.0007:
            return None, None
        real_coeffs = np.polyfit(line[1]*self.x_m_per_pix, line[0]*self.y_m_per_pix, 2)
        return coeffs, real_coeffs

    def _draw_lines(self, img: Mat, left_line, right_line) -> Mat:
        # if lines is None:
        #     return img
        # left_line, right_line = self._fit_lane_lines(left_line, right_line)
        img = np.copy(img)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        half_height = img.shape[0] // 2
        # y = np.linspace(half_height + 85, img.shape[0] -1, img.shape[0] - half_height)
        y = np.linspace(0, img.shape[0] - 1, img.shape[0])
        if left_line is not None:
            if len(left_line) == 2:
                x_left = left_line[0] * y + left_line[1]
            elif len(left_line) == 3:
                x_left = left_line[0] * y ** 2 + left_line[1] * y + left_line[2]
            left_points = np.array(np.transpose(np.vstack([x_left, y])), np.int32)
            cv.polylines(line_img, [left_points], False, (255, 0, 0), thickness=40)

        if right_line is not None:
            if len(right_line) == 2:
                x_right = right_line[0] * y + right_line[1]
            elif len(right_line) == 3:
                x_right = right_line[0] * y ** 2 + right_line[1] * y + right_line[2]
            right_points = np.array(np.transpose(np.vstack([x_right, y])), np.int32)
            cv.polylines(line_img, [right_points], False, (0, 0, 255), thickness=40)
        line_img = self.calibration.warp_from_birdseye(line_img)
        return cv.addWeighted(img, 0.6, line_img, 1, 0)

    def seperate_lines_on_thresh(self, img: Mat):
        middle = self.width // 2
        middle_left = middle - self.width // 8
        middle_right = middle + self.width // 8
        right_y, right_x = np.where(img[:, middle_right:] == 255)
        left_y, left_x = np.where(img[:, :middle_left] == 255)
        right_x += middle_right
        return np.array([left_x, left_y]), np.array([right_x, right_y])

    def _calculate_curvature(self, fit_left_line, fit_right_line):
        if fit_left_line is None or fit_right_line is None:
            return -1
        a_left, b_left, c_left = fit_left_line
        a_right, b_right, c_right = fit_right_line
        y = self.height - 1
        y_eval = self.height / 2 # * self.y_m_per_pix
        # x_eval = (self.width / 2) * self.x_m_per_pix
        left_curvature = ((1 + (2 * a_left * y_eval + b_left) ** 2) ** 1.5) / np.absolute(2 * a_left)
        right_curvature = ((1 + (2 * a_right * y_eval + b_right) ** 2) ** 1.5) / np.absolute(2 * a_right)
        mean_curvature = np.mean([left_curvature, right_curvature])
        return mean_curvature

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
