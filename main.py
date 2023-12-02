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
        video = cv.VideoCapture('./images/Udacity/challenge_video.mp4')
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            frame = self.detect_lane(frame)
            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        video.release()
        cv.destroyAllWindows()

        # img = cv.imread('./images/Udacity/image001.jpg')
        # imgs = [cv.imread(f'./images/Udacity/image00{i}.jpg') for i in range(1, 7)]
        # # img = self.detect_lane(img)
        # imgs = [self.detect_lane(img) for img in imgs]
        # for k, img in enumerate(imgs):
        #     cv.imshow(f'img{k+1}', img)
        #     cv.waitKey(0)
        #     cv.destroyAllWindows()

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

    def _draw_lines(self, img: Mat, lines: np.ndarray) -> Mat:
        if lines is None:
            return img
        img = np.copy(img)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)
        img = cv.addWeighted(img, 0.8, line_img, 1, 0)
        return line_img


if __name__ == '__main__':
    LaneDetection().run()
