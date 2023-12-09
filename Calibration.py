import typing
import cv2 as cv
import numpy as np
from typing import Sequence
from cv2.typing import MatLike as Mat
import matplotlib.pyplot as plt
import pickle
import os


class Calibration:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super().__new__(cls)
            cls.__instance.__init__(*args, **kwargs)
            cls.__init__ = lambda *aargs, **kwaargs: None
        return cls.__instance

    def __init__(self,
                 calib_path: str = './images/Udacity/calib/',
                 calib_config_path: str = './images/Udacity/calib/config.pickle',
                 warp_mat_path: str = './images/Udacity/calib/warp_mat.pickle'):
        self.dist: Mat | None = None
        self.mtx: Mat | None = None
        self.roi: Mat | None = None
        self.newcammat: Mat | None = None
        self.warp_src: Mat | None = None
        self.warp_dst: Mat | None = None
        self.own_dir = os.path.dirname(os.path.abspath(__file__))
        self.calib_path: str = os.path.join(self.own_dir, calib_path)
        self.calib_config_path = os.path.join(self.own_dir, calib_config_path)
        self._get_trans_mat_pickle(os.path.join(self.own_dir, warp_mat_path))
        self._calibrate()

    def _get_obj_img_points(self) -> tuple[Sequence[Mat], Sequence[Mat]]:
        """
        Gets the object and image points for the calibration.
        :return: Object and image points
        """
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

    def _calibrate(self) -> None:
        """
        Calibrates the camera and saves parameters in a pickle file.
        :return: None
        """
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
        """
        Undistorts the image.
        :param img: Image to undistort
        :return: Undistorted image
        """
        undist_img = cv.undistort(img, self.mtx, self.dist, None, self.newcammat)
        x, y, w, h = self.roi
        undist_img = undist_img[y: y+h, x:x+w]
        return undist_img

    # def _get_trans_mat_pickle(self, warp_mat_path: str):
    #     # self.warp_src, self.warp_dst = pickle.load(open(warp_mat_path, 'rb'))
    #     return

    def warp_to_birdseye(self, img: Mat) -> Mat:
        """
        Warps the image to birdseye view.
        :param img: Image to warp
        :return: Warped image
        """
        h, w = img.shape[:2]
        middle_x = img.shape[1] / 2
        middle_y = img.shape[0] / 2
        top_left = (middle_x - 140, middle_y + 100)
        top_right = (middle_x + 140, middle_y + 100)
        triangle_left = (img.shape[1] / 20, img.shape[0])
        triangle_right = (img.shape[1] * 95 / 100, img.shape[0])
        vertices = np.array([[top_left, top_right, triangle_right, triangle_left]], dtype=np.int32)
        roi = np.array([[[530, 100], [740, 100], [1080, 596], [260, 596]]], dtype=np.int32)
        dst = np.array([[[300, 0], [980, 0], [980, 720], [300, 720]]], dtype=np.int32)
        dst = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.int32)
        self.warp_src = vertices
        self.warp_dst = dst
        M = cv.getPerspectiveTransform(np.float32(self.warp_src), np.float32(self.warp_dst))
        warped = cv.warpPerspective(img, M, (w, h), flags=cv.INTER_LINEAR)
        return warped

    def warp_from_birdseye(self, img: Mat) -> Mat:
        """
        Warps the image from birdseye view to the original perspective.
        :param img: Image to warp
        :return: warped image
        """
        h, w = img.shape[:2]
        M = cv.getPerspectiveTransform(np.float32(self.warp_dst), np.float32(self.warp_src))
        warped = cv.warpPerspective(img, M, (w, h), flags=cv.INTER_LINEAR)
        return warped

    def _get_trans_mat_pickle(self, warp_mat_path: str) -> None:
        """
        Gets the transformation matrices from a pickle file.
        :param warp_mat_path: Path to the pickle file
        :return: None
        """
        if os.path.exists(warp_mat_path):
            self.warp_src, self.warp_dst = pickle.load(open(warp_mat_path, 'rb'))
            print(f"{self.warp_src}\n{self.warp_dst}")
            return


if __name__ == '__main__':
    calibration = Calibration()
    image = calibration.undistort(cv.cvtColor(cv.imread('./images/Udacity/image001.jpg'), cv.COLOR_BGR2RGB))
    # image = calibration.warp_to_birdseye(image)
    plt.figure(figsize=(30,30))
    plt.imshow(image)
    plt.show()
