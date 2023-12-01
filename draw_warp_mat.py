import cv2 as cv
import numpy as np
from Calibration import Calibration
import pickle

"""
Dieses Skript dient dazu, die Punkte für die Transformationsmatrix zu bestimmen.
Dazu wird ein Bild ausgewählt und mit der Maus die Eckpunkte des Straßenbereichs markiert.
Die Punkte werden in der Konsole ausgegeben und können dann in der main.py in die 
Transformationsmatrix eingetragen werden.
Output:
    (578, 393)
    (654, 394)
    (960, 588)
    (296, 585)
    Y-Koordinate Glätten:
    (578, 394)
    (654, 394)
    (960, 588)
    (296, 588)
    
"""

def draw_circle(event, x, y, flags, param):
    global img, scanner_points
    if event == cv.EVENT_LBUTTONDBLCLK:
        scanner_points.append((x, y))
        print(f"({x}, {y})")
    for i in range(len(scanner_points)):
        cv.circle(img, (scanner_points[i][0], scanner_points[i][1]), 5, (0, 0, 255), -1)


if __name__ == "__main__":
    src = np.matrix([[578, 393], [654, 394], [960, 588], [296, 585]])
    dst = np.matrix([[300, 0], [980, 0], [980, 720], [300, 720]])

    with open('./images/Udacity/calib/warp_mat.pickle', 'wb') as f:
        pickle.dump((src, dst), f)
    img_scanner = cv.imread(
        'C:/users/inf21034/PycharmProjects/conventional_lane_detection/images/Udacity/image001.jpg')
    img_scanner = Calibration().undistort(img_scanner)
    img = img_scanner.copy()
    scanner_points = []
    cv.namedWindow('image')
    cv.setMouseCallback('image', draw_circle)
    while True:
        cv.imshow('image', img)
        if cv.waitKey(20) & 0xFF == '27':
            break
    cv.destroyAllWindows()
    print(scanner_points)
