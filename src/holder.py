# import required libraries
import os
import numpy as np #type: ignore
import cv2 #type: ignore
from matplotlib import pyplot as plt #type: ignore

class DepthMap:
    def __init__(self,imgL,imgR,calibFile):
        self.imgL = imgL
        self.imgR = imgR
        self.calibFile = calibFile

    def CalibrateDepthMap(self):
        
        return 

    def DepthMapBM(self):
        nDispFactor = 6
        # Initiate and StereoBM object
        stereo = cv2.StereoBM.create(numDisparities=16, blockSize=15)

        # compute the disparity map
        disparity = stereo.compute(self.imgL,self.imgR)

        #return the created depth map
        return disparity
    
    def DepthMapSGBMTest1(self):
        window_size = 7
        min_disp = 16
        nDispFactor = 17
        num_disp = 16*nDispFactor-min_disp

        stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                    numDisparities = num_disp,
                                    blockSize = window_size,
                                    P1 = 8*3*window_size**2,
                                    P2 = 32*3*window_size**2,
                                    disp12MaxDiff=1,
                                    uniquenessRatio=15,
                                    speckleWindowSize=0,
                                    speckleRange=2,
                                    preFilterCap=63,
                                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        disparity = stereo.compute(self.imgL,self.imgR).astype(np.
        float32) / 16.0

        return disparity
    
    def DepthMapSGBMTest2(self):
        # Calibration parameters
        cam0 = np.array([[1734.04, 0, -133.21],
                        [0, 1734.04, 542.27],
                        [0, 0, 1]])

        cam1 = np.array([[1734.04, 0, -133.21],
                        [0, 1734.04, 542.27],
                        [0, 0, 1]])

        doffs = 0
        baseline = 529.50
        width = 1920
        height = 1080
        ndisp = 190
        vmin = 55
        vmax = 160

        # Create StereoSGBM object
        stereo = cv2.StereoSGBM_create(
            minDisparity=vmin,
            numDisparities=vmax - vmin,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Compute the disparity map
        disparity_map = stereo.compute(self.imgL, self.imgR).astype(np.float32) / 16.0

        return disparity_map
    
    def DepthMapSGBMTest3(self):
        #Calibration parameters
        cam0 = np.array([[1734.04, 0, -133.21],
                        [0, 1734.04, 542.27],
                        [0, 0, 1]])

        cam1 = np.array([[1734.04, 0, -133.21],
                        [0, 1734.04, 542.27],
                        [0, 0, 1]])

        doffs = 0
        baseline = 529.50
        width = 1920
        height = 1080
        ndisp = 190
        vmin = 55
        vmax = 160

        # Example rotation and translation matrices (need actual values for real rectification)
        R = np.eye(3)  # Placeholder, should be the rotation matrix from stereo calibration
        T = np.array([baseline, 0, 0])  # Placeholder, should be the translation vector from stereo calibration

        # Rectify the images (assuming rectification is required and R, T are available)
        # Stereo rectify
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cam0, None, cam1, None, (width, height), R, T, alpha=0)

        # Compute the rectification maps
        map1x, map1y = cv2.initUndistortRectifyMap(cam0, None, R1, P1, (width, height), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(cam1, None, R2, P2, (width, height), cv2.CV_32FC1)

        # Apply the rectification maps
        rectified_left = cv2.remap(self.imgL, map1x, map1y, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(self.imgR, map2x, map2y, cv2.INTER_LINEAR)

        # Create StereoSGBM object
        stereo = cv2.StereoSGBM_create(
            minDisparity=vmin,
            numDisparities=vmax - vmin,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Compute the disparity map
        disparity_map = stereo.compute(rectified_left, rectified_right).astype(np.float32) / 16.0

        return disparity_map
    
    def DepthMapSGBMTest4(self):
        # Calibration parameters
        cam0 = np.array([[1734.04, 0, -133.21],
                        [0, 1734.04, 542.27],
                        [0, 0, 1]])

        cam1 = np.array([[1734.04, 0, -133.21],
                        [0, 1734.04, 542.27],
                        [0, 0, 1]])

        doffs = 0
        baseline = 529.50
        width = 1920
        height = 1080
        ndisp = 190
        vmin = 55
        vmax = 160

        # Create StereoSGBM object with fine-tuned parameters
        stereo = cv2.StereoSGBM_create(
            minDisparity=vmin,
            numDisparities=vmax - vmin,
            blockSize=7,
            P1=8 * 3 * 7**2,
            P2=32 * 3 * 7**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Compute the disparity map
        disparity_map = stereo.compute(self.imgL, self.imgR).astype(np.float32) / 16.0

        # Post-processing to improve disparity map quality
        disparity_map = cv2.medianBlur(disparity_map, 5)

        # Apply disparity WLS filter (requires the ximgproc module from OpenCV contrib)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
        right_matcher = cv2.ximgproc.createRightMatcher(stereo)
        disparity_right = right_matcher.compute(self.imgR, self.imgL).astype(np.float32) / 16.0

        wls_filter.setLambda(8000.0)
        wls_filter.setSigmaColor(1.5)
        filtered_disparity = wls_filter.filter(disparity_map, self.imgL, disparity_map_right=disparity_right)

        # Normalize the disparity map for visualization
        disparity_map_normalized = cv2.normalize(filtered_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disparity_map_normalized = np.uint8(disparity_map_normalized)

        return disparity_map_normalized
    
    def DepthMapSGBMTest5(self):
        # Calibration parameters
        cam0 = np.array([[1734.04, 0, -133.21],
                        [0, 1734.04, 542.27],
                        [0, 0, 1]])

        cam1 = np.array([[1734.04, 0, -133.21],
                        [0, 1734.04, 542.27],
                        [0, 0, 1]])

        doffs = 0
        baseline = 529.50
        width = 1920
        height = 1080
        ndisp = 190
        vmin = 55
        vmax = 160
        
        # Load stereo images (assumed to be rectified)
        left_image = self.imgL
        right_image = self.imgR

        # Apply histogram equalization
        left_image = cv2.equalizeHist(left_image)
        right_image = cv2.equalizeHist(right_image)

        # Apply Laplacian filter to enhance edges
        left_image = cv2.Laplacian(left_image, cv2.CV_8U)
        right_image = cv2.Laplacian(right_image, cv2.CV_8U)

        # Create StereoSGBM object with fine-tuned parameters
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=ndisp,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1, # set to 1 or 2
            uniquenessRatio=5, # 
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63, # set between 5 and 63
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Compute the disparity map
        disparity_map = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
        # Post-processing to improve disparity map quality
        disparity_map = cv2.medianBlur(disparity_map, 5)

        # Apply bilateral filter to smooth the disparity map while preserving edges
        disparity_map = cv2.bilateralFilter(disparity_map, 9, 75, 75)
      
        # Apply disparity WLS filter (requires the ximgproc module from OpenCV contrib)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
        right_matcher = cv2.ximgproc.createRightMatcher(stereo)
        disparity_right = right_matcher.compute(right_image, left_image).astype(np.float32) / 16.0

        wls_filter.setLambda(8000.0)
        wls_filter.setSigmaColor(1.5)
        filtered_disparity = wls_filter.filter(disparity_map, left_image, disparity_map_right=disparity_right)

        return filtered_disparity

        # Normalize the disparity map for visualization
        disparity_map_normalized = np.uint8(filtered_disparity)

    def DepthMapSGBMTest6(self):
        #Calibration parameters
        cam0 = np.array([[1734.04, 0, -133.21],
                        [0, 1734.04, 542.27],
                        [0, 0, 1]])

        cam1 = np.array([[1734.04, 0, -133.21],
                        [0, 1734.04, 542.27],
                        [0, 0, 1]])

        doffs = 0
        baseline = 529.50
        width = 1920
        height = 1080
        ndisp = 190
        vmin = 55
        vmax = 160

        # Example rotation and translation matrices (need actual values for real rectification)
        R = np.eye(3)  # Placeholder, should be the rotation matrix from stereo calibration
        T = np.array([baseline, 0, 0])  # Placeholder, should be the translation vector from stereo calibration

        # Rectify the images (assuming rectification is required and R, T are available)
        # Stereo rectify
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cam0, None, cam1, None, (width, height), R, T, alpha=0)

        # Compute the rectification maps
        map1x, map1y = cv2.initUndistortRectifyMap(cam0, None, R1, P1, (width, height), cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(cam1, None, R2, P2, (width, height), cv2.CV_32FC1)

        # Apply the rectification maps
        rectified_left = cv2.remap(self.imgL, map1x, map1y, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(self.imgR, map2x, map2y, cv2.INTER_LINEAR)

        # Create StereoSGBM object with fine-tuned parameters
        stereo = cv2.StereoSGBM_create(
            minDisparity=vmin,
            numDisparities = vmax - vmin,
            blockSize=7,
            P1=8 * 3 * 7**2,
            P2=32 * 3 * 7**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Compute the disparity map
        disparity_map = stereo.compute(rectified_left, rectified_right).astype(np.float32) / 16.0

        # Post-processing to improve disparity map quality
        disparity_map = cv2.medianBlur(disparity_map, 5)

        # Apply bilateral filter to smooth the disparity map while preserving edges
        disparity_map = cv2.bilateralFilter(disparity_map, 9, 75, 75)
      
        # Apply disparity WLS filter (requires the ximgproc module from OpenCV contrib)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
        right_matcher = cv2.ximgproc.createRightMatcher(stereo)
        disparity_right = right_matcher.compute(rectified_right, rectified_left).astype(np.float32) / 16.0

        wls_filter.setLambda(8000.0)
        wls_filter.setSigmaColor(1.5)
        filtered_disparity = wls_filter.filter(disparity_map, rectified_left, disparity_map_right=disparity_right)

        disparity_map_normalized = cv2.normalize(filtered_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disparity_map_normalized = np.uint8(disparity_map_normalized)

        return disparity_map_normalized
    
    
def testing():
    f = "artroom2"
    # set path for read and write
    dir = str(f)
    path = "./data/%s/" % dir

    # read two input images as grayscale images
    imgL = cv2.imread('%sim0.png' % path, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread('%sim1.png' % path, cv2.IMREAD_GRAYSCALE)
    calibFile = open("%scalib.txt" % path ,'r')

    # call the function
    dp = DepthMap(imgL,imgR,calibFile)
    disparity = dp.DepthMapSGBMTest6()

    # save the returned depth map
    cv2.imwrite('./depth_maps/%s.png' % dir, disparity)

testing()