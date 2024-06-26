import cv2 #type: ignore
import numpy as np #type: ignore

class Rectify:
    def __init__(self, cal, imgL, imgR):
        self.imgL = imgL
        self.imgR = imgR
        self.cal = cal
        self.RectifyImages()
        # self.Histogram()
        # self.Laplacian()

    def RectifyImages(self):
        cal = self.cal

        # Example rotation and translation matrices (need actual values for real rectification)
        R = np.eye(3)  # Placeholder, should be the rotation matrix from stereo calibration
        T = np.array([cal.get_baseline(), 0, 0])  # Placeholder, should be the translation vector from stereo calibration

        # Rectify the images (assuming rectification is required and R, T are available)
        # Stereo rectify
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cal.get_cam0(), None, cal.get_cam1(), None, (cal.get_width(), cal.get_height()), R, T, alpha=0)

        # Compute the rectification maps
        map1x, map1y = cv2.initUndistortRectifyMap(cal.get_cam0(), None, R1, P1, (cal.get_width(), cal.get_height()), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(cal.get_cam1(), None, R2, P2, (cal.get_width(), cal.get_height()), cv2.CV_32FC1)
                                                   
        # Apply the rectification maps
        self.imgL = cv2.remap(self.imgL, map1x, map1y, cv2.INTER_LINEAR)
        self.imgR = cv2.remap(self.imgR, map2x, map2y, cv2.INTER_LINEAR)
    
    def Histogram(self):
        # Apply histogram equalization
        self.imgL = cv2.equalizeHist(self.imgL)
        self.imgR = cv2.equalizeHist(self.imgR)

    def Laplacian(self):
        # Apply Laplacian filter to enhance edges
        self.imgL = cv2.Laplacian(self.imgL, cv2.CV_8U)
        self.imgR = cv2.Laplacian(self.imgR, cv2.CV_8U)

    def get_imgL(self):
        return self.imgL
    
    def get_imgR(self):
        return self.imgR