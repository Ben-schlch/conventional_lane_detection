from Calibration import Calibration
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from cv2.typing import MatLike as Mat
from preprocess import preprocess
from time import time
from concurrent.futures import ThreadPoolExecutor
import threading
import queue


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
        self.image_queue = queue.Queue()
        self.frames = 0
        pass

    def run_with_batch(self):
        start_time = time()
        producer = threading.Thread(target=self._producer, args=('./images/Udacity/project_video.mp4',))
        consumer = threading.Thread(target=self._consumer, args=(8,))
        producer.start()
        consumer.start()
        producer.join()
        consumer.join()
        end_time = time()
        print(f"Frames: {self.frames}")
        print(f"Time: {end_time - start_time}")
        print(f"FPS: {self.frames / (end_time - start_time)}")
        cv.destroyAllWindows()

    def run(self):
        start_time = time()
        video = cv.VideoCapture('./images/Udacity/project_video.mp4')
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame = self.detect_lane(frame)
            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        end_time = time()
        print(f"Frames: {self.frames}")
        print(f"Time: {end_time - start_time}")
        print(f"FPS: {self.frames / (end_time - start_time)}")

    def _producer(self, path):
        video = cv.VideoCapture(path)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                self.image_queue.put(-1)
                break
            self.image_queue.put(frame)
        video.release()

    def _consumer(self, batch_size=8):
        video_over = False
        while not video_over:
            print("cons")
            batch_frames = []
            for _ in range(batch_size):
                frame = self.image_queue.get()
                if frame is -1:
                    video_over = True
                    break
                batch_frames.append(frame)
                self.frames += 1
                self.image_queue.task_done()
            if len(batch_frames) == 0:
                if video_over:
                    break
                continue

            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                results = executor.map(self.detect_lane, batch_frames)
                for result in results:
                    cv.imshow('frame', result)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
        print("done")
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
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
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
        return left_line_coeffs, right_line_coeffs, real_left_coeffs, real_right_coeffs

    def __fit_line(self, line: np.ndarray | None):
        if line is None:
            return None, None
        if 0 in line.shape[:2]:
            return None, None
        if len(line[0]) < 3:
            return None, None
        coeffs = np.polyfit(line[1], line[0], 2)
        if abs(coeffs[0]) > 0.0007:  # Unrealistische Kurvenverläufe verwerfen
            return None, None
        real_coeffs = np.polyfit(line[0] * self.x_m_per_pix, line[1] * self.y_m_per_pix, 2)
        return coeffs, real_coeffs

    def _draw_lines(self, img: Mat, left_line, right_line) -> Mat:
        img = np.copy(img)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        half_height = img.shape[0] // 2
        # y = np.linspace(half_height + 85, img.shape[0] -1, img.shape[0] - half_height)
        y = np.linspace(0, img.shape[0] - 1, img.shape[0])
        if left_line is not None:
            left_points = self.__draw_line(left_line, y)
            cv.polylines(line_img, [left_points], False, (255, 0, 0), thickness=40)
        if right_line is not None:
            right_points = self.__draw_line(right_line, y)
            cv.polylines(line_img, [right_points], False, (0, 0, 255), thickness=40)
        line_img = self.calibration.warp_from_birdseye(line_img)
        return cv.addWeighted(img, 0.6, line_img, 1, 0)

    def __draw_line(self, line, y):
        if len(line) == 2:
            x = line[0] * y + line[1]
        elif len(line) == 3:
            x = line[0] * y ** 2 + line[1] * y + line[2]
        points = np.array(np.transpose(np.vstack([x, y])), np.int32)
        return points

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
        x_eval = (self.width / 2) * self.x_m_per_pix
        left_curvature = ((1 + (2 * a_left * x_eval + b_left) ** 2) ** 1.5) / np.absolute(2 * a_left)
        right_curvature = ((1 + (2 * a_right * x_eval + b_right) ** 2) ** 1.5) / np.absolute(2 * a_right)
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
