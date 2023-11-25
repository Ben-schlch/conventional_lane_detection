import typing
import cv2 as cv
import numpy as np
from typing import Sequence
from cv2.typing import MatLike as Mat
import matplotlib.pyplot as plt
import pickle
import os


class Calibration:

    def __init__(self,
                 calib_path: str = './images/Udacity/calib/',
                 calib_config_path: str = './images/Udacity/calib/config.pickle'):
        self.dist: Mat | None = None
        self.mtx: Mat | None = None
        self.roi: Mat | None = None
        self.newcammat: Mat | None = None
        self.calib_path: str = calib_path
        self.calib_config_path = calib_config_path
        self._calibrate()

    def _get_obj_img_points(self) -> tuple[Sequence[Mat], Sequence[Mat]]:
        grid_size = (9, 6)
        calib_imgs = [cv.cvtColor(cv.imread(f'{self.calib_path}calibration{i}.jpg'), cv.COLOR_BGR2RGB) for i in
                      range(1, 21)]
        undistorted_imgs = []
        t_cap_prev = 0
        objpoints = []
        imgpoints = []
        obj_p = np.zeros((6 * 9, 3), np.float32)
        obj_p[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        for k in range(len(calib_imgs)):
            calib_img = calib_imgs[k]
            gray_calib_img = cv.cvtColor(calib_img, cv.COLOR_RGB2GRAY)

            ret, corners = cv.findChessboardCorners(gray_calib_img, grid_size, None)

            if ret:
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_subpix = cv.cornerSubPix(gray_calib_img, corners=corners, winSize=(11, 11), zeroZone=(-1, 1),
                                                 criteria=criteria)

                imgpoints.append(corners_subpix)
                objpoints.append(obj_p)

        return objpoints, imgpoints

    def _calibrate(self):
        if os.path.exists(self.calib_config_path):
            with open(self.calib_config_path, "rb") as f:
                self.mtx, self.dist, self.roi, self.newcammat = pickle.load(f)
            return
        img_path = self.calib_path + 'calibration1.jpg'
        img = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)
        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        objpoints, imgpoints = self._get_obj_img_points()
        ret, self.mtx, self.dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray_img.shape[::-1],
                                                                    None,None)
        h, w = img.shape[:2]
        self.newcammat, self.roi = cv.getOptimalNewCameraMatrix(self.mtx,
                                                                self.dist,
                                                                (w, h),
                                                                1,
                                                                (w, h))
        # Konfiguration Serialisieren
        with open(self.calib_config_path, "wb") as f:
            pickle.dump((self.mtx, self.dist, self.roi, self.newcammat), f)

    def undistort(self, img: Mat) -> Mat:
        undist_img = cv.undistort(img, self.mtx, self.dist, None, self.newcammat)
        x, y, w, h = self.roi
        undist_img = undist_img[y: y+h, x:x+w]
        return undist_img


if __name__ == '__main__':
    calibration = Calibration()
    image = calibration.undistort(cv.cvtColor(cv.imread('./images/image001.jpg'), cv.COLOR_BGR2RGB))
    plt.figure(figsize=(30,30))
    plt.imshow(image)
    plt.show()
