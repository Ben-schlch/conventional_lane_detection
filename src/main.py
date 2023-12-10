import sys

from src.Calibration import Calibration
import cv2 as cv
import numpy as np
from cv2.typing import MatLike as Mat
from src.preprocess import preprocess
from time import time
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import argparse


class LaneDetection:

    def __init__(self, visualize=False, video='./images/Udacity/project_video.mp4'):
        self.calibration = Calibration()
        self.previous_left_line = None
        self.previous_right_line = None
        self.previous_real_left = None
        self.previous_real_right = None
        self.width = 0
        self.height = 0
        self.y_m_per_pix = 10 / 720
        self.x_m_per_pix = 3.7 / 300
        self.image_queue = queue.Queue()
        self.frames = 0
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.visualize = visualize
        self.video = video
        pass

    def run_with_batch(self) -> None:
        """
        Runs the lane detection with batch size of 8. Measures FPS and prints it.
        :return: None
        """
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

    def run(self) -> None:
        """
        Runs the lane detection with batch size of 1. Measures FPS and prints it.
        :return: None
        """
        start_time = time()
        consumer = threading.Thread(target=self._consumer, args=(1,))
        producer = threading.Thread(target=self._producer, args=(self.video,))
        producer.start()
        consumer.start()
        producer.join()
        consumer.join()

        end_time = time()
        print(f"Frames: {self.frames}")
        print(f"Time: {end_time - start_time}")
        print(f"FPS: {self.frames / (end_time - start_time)}")

    def _producer(self, path) -> None:
        """
        Produces images from a video file and puts them into a queue
        :param path: Path to the video file
        :return: None
        """
        video = cv.VideoCapture(path)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                self.image_queue.put(-1)
                break
            self.image_queue.put(frame)
        video.release()

    def _consumer(self, batch_size=8) -> None:
        """
        Consumes images from a queue and processes them. If batch_size is 1, the images are processed sequentially.
        :param batch_size: Number of frames to process in parallel.
        :return: None
        """
        next_time = time()
        prev = next_time
        prev_fps = 0
        if batch_size == 1:
            while True:
                frame = self.image_queue.get()
                if frame is -1:
                    break
                self.frames += 1
                self.image_queue.task_done()
                res = self.detect_lane(frame)
                if self.visualize:
                    if self.frames % 5 == 0:
                        prev = next_time
                        next_time = time()
                        prev_fps = 5 / (next_time - prev)
                    cv.putText(res, f"FPS: {prev_fps:.2f}",
                               (1000, 50), self.font, 1, (255, 0, 255), 2, cv.LINE_AA)
                    cv.imshow("frame", res)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
            if self.visualize:
                cv.destroyAllWindows()
            return
        video_over = False

        while not video_over:
            batch_frames = []
            for _ in range(batch_size):
                frame = self.image_queue.get()
                if frame is -1:
                    video_over = True
                    break
                batch_frames.append(frame)
                self.image_queue.task_done()
            if len(batch_frames) == 0:
                if video_over:
                    break
                continue
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                results = executor.map(self.detect_lane, batch_frames)
                for result in results:
                    self.frames += 1
                    cv.imshow('frame', result)
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break
        print("done")

        cv.destroyAllWindows()

    def detect_lane(self, img: Mat) -> Mat:
        """
        Detects the lane in an image.
        :param img: Frame to detect the lane in.
        :return: Image with lanes drawn on it.
        """
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = self.calibration.undistort(img)
        img_processed = preprocess(img)
        self.width = img.shape[1]
        left_line, right_line = self.seperate_lines_on_thresh(img_processed)
        fit_left_line, fit_right_line, real_left, real_right = self._fit_lane_lines(right_line, left_line)
        radius = self._calculate_curvature(real_left, real_right)
        if self.visualize:
            img = self._draw_lines(img, fit_left_line, fit_right_line)
            cv.putText(img, f"Radius: {radius:.2f}m", (10, 50), self.font, 1, (255, 255, 255), 2, cv.LINE_AA)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        return img

    def _fit_lane_lines(self, right_line: np.ndarray, left_line: np.ndarray) \
            -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """
        Fits a polynomial to the lane lines and a real polynomial in meters.
        :param right_line: Array of points of the right lane line.
        :param left_line: Array of points of the left lane line.
        :return: Polynomial of the left lane line, polynomial of the right lane line, real polynomial of the left lane
        """
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

    def __fit_line(self, line: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Fits a polynomial to a line.
        :param line: Array of points of the line.
        :return: Coefficients and 'real' coefficients of the polynomial. If the line is None, None is returned.
        """
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

    def _draw_lines(self, img: Mat, left_line: np.ndarray, right_line: np.ndarray) -> Mat:
        """
        Draws the lane lines on an image.
        :param img: Frame to draw the lines on.
        :param left_line: Polynomial of the left lane line.
        :param right_line: Polynomial of the right lane line.
        :return: Image with the lane lines drawn on it. Warped back to the original perspective.
        """
        img = np.copy(img)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        y = np.linspace(0, img.shape[0] - 1, img.shape[0])
        line_img = self.__draw_line(right_line, y, (0, 0, 255), line_img)
        line_img = self.__draw_line(left_line, y, (255, 0, 0), line_img)
        line_img = self.calibration.warp_from_birdseye(line_img)
        return cv.addWeighted(img, 0.6, line_img, 1, 0)

    def __draw_line(self, line: np.ndarray, y: np.ndarray, color: tuple[int, int, int], line_img: Mat) -> Mat:
        """
        Draws a line on an image.
        :param line: Coefficients of the polynomial.
        :param y: Linspace of y values.
        :param color: RGB Color of the line.
        :param line_img: Image to draw the line on.
        :return: Image with the line drawn on it.
        """
        if line is None:
            return line_img
        if len(line) == 2:
            x = line[0] * y + line[1]
        elif len(line) == 3:
            x = line[0] * y ** 2 + line[1] * y + line[2]
        points = np.array(np.transpose(np.vstack([x, y])), np.int32)
        cv.polylines(line_img, [points], False, color, thickness=40)
        return line_img

    def seperate_lines_on_thresh(self, img: Mat) -> tuple[np.ndarray, np.ndarray]:
        """
        Seperates the image into two parts, one for the left lane line and one for the right lane line.
        :param img: Image to seperate.
        :return: left lane points, right lane points
        """
        middle = self.width // 2
        middle_left = middle - self.width // 8
        middle_right = middle + self.width // 8
        right_y, right_x = np.where(img[:, middle_right:] == 255)
        left_y, left_x = np.where(img[:, :middle_left] == 255)
        right_x += middle_right
        return np.array([left_x, left_y]), np.array([right_x, right_y])

    def _calculate_curvature(self, fit_left_line, fit_right_line) -> float:
        """
        Calculates the curvature of the lane.
        :param fit_left_line: 'Real' Coefficients of the polynomial of the left lane line.
        :param fit_right_line: 'Real' Coefficients of the polynomial of the right lane line.
        :return: Curve Radius in meters.
        """
        if fit_left_line is None or fit_right_line is None:
            return -1
        a_left, b_left, c_left = fit_left_line
        a_right, b_right, c_right = fit_right_line
        x_eval = (self.width / 2) * self.x_m_per_pix
        left_curvature = ((1 + (2 * a_left * x_eval + b_left) ** 2) ** 1.5) / np.absolute(2 * a_left)
        right_curvature = ((1 + (2 * a_right * x_eval + b_right) ** 2) ** 1.5) / np.absolute(2 * a_right)
        mean_curvature = np.mean([left_curvature, right_curvature])
        return mean_curvature


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lane Detection')
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.add_argument('--video', dest='video', action="store")
    parser.set_defaults(visualize=False, video='./images/Udacity/project_video.mp4')
    args = parser.parse_args()
    LaneDetection(args.visualize, args.video).run()
