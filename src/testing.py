# import required libraries
import os
import cv2 #type: ignore
from camera import Calibration
from rectify import Rectify
from depth_map import DepthMap

def testing():
    for f in os.listdir("data"):
        if os.path.isdir(f'./data/{f}'):
            print(f)
            # set path for read and write
            dir = str(f)
            path = "./data/%s/" % dir

            # read two input images as grayscale images
            imgL = cv2.imread('%sim0.png' % path, cv2.IMREAD_GRAYSCALE)
            imgR = cv2.imread('%sim1.png' % path, cv2.IMREAD_GRAYSCALE)
            calibFile = "%scalib.txt" % path

            # call the functions
            cal = Calibration(calibFile)
            rect = Rectify(cal, imgL, imgR)
            dp = DepthMap(rect, cal)

            # save the returned depth map
            disparity = dp.get_disparity_map()
            cv2.imwrite('./depth_maps/%s.png' % dir, disparity)

testing()