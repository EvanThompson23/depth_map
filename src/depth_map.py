# import required libraries
import cv2 #type: ignore
import numpy as np #type: ignore

class DepthMap:
    def __init__(self, rect, cal):
        self.rect = rect
        self.cal = cal
        self.disparity_Map = None
        self.stereo = cv2.StereoSGBM_create()
        self.DepthMap()
        self.PostProcessing()

    def DepthMap(self):
        cal = self.cal

        fx = cal.get_cam0()[0,0]
        pixel_aspect_ratio = fx * cal.get_baseline() / cal.get_ndisp()
        blockSize = 7 # adjust between 3 and 11
        P1= (8 * 3 * blockSize **2) /pixel_aspect_ratio
        P2= (32 * 3 * blockSize **2) /pixel_aspect_ratio
        disp12MaxDiff = 1 # set to 1 or 2 
        uniquenessRatio = 11 # set between 5 and 15
        speckleWindowSize = 100
        speckleRange = 32
        preFilterCap = 30 # set between 5 and 63

        self.stereo.setMinDisparity(cal.get_vmin())
        self.stereo.setNumDisparities(cal.get_ndisp())
        self.stereo.setBlockSize(blockSize)
        self.stereo.setP1(round(P1))
        self.stereo.setP2(round(P2))
        self.stereo.setDisp12MaxDiff(disp12MaxDiff)
        self.stereo.setPreFilterCap(preFilterCap)
        self.stereo.setUniquenessRatio(uniquenessRatio)
        self.stereo.setSpeckleWindowSize(speckleWindowSize)
        self.stereo.setSpeckleRange(speckleRange)
        self.stereo.setMode(cv2.STEREO_SGBM_MODE_SGBM_3WAY)

        self.set_disparity_map(self.stereo.compute(self.rect.get_imgL(), self.rect.get_imgR()).astype(np.float32) / 16.0)

    def PostProcessing(self):

        # Post-processing to improve disparity map quality
        disparity_map = cv2.medianBlur(self.get_disparity_map(), 1)

        # Apply bilateral filter to smooth the disparity map while preserving edges
        disparity_map = cv2.bilateralFilter(disparity_map, 9, 75, 75)
      
        # Apply disparity WLS filter (requires the ximgproc module from OpenCV contrib)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.get_stereo())
        right_matcher = cv2.ximgproc.createRightMatcher(self.get_stereo())
        disparity_right = right_matcher.compute(self.rect.get_imgR(), self.rect.get_imgL()).astype(np.float32) / 16.0

        wls_filter.setLambda(8000.0)
        wls_filter.setSigmaColor(1.5)
        filtered_disparity = wls_filter.filter(disparity_map, self.rect.get_imgL(), disparity_map_right=disparity_right)

        disparity_map_normalized = cv2.normalize(filtered_disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        self.set_disparity_map(np.uint8(disparity_map_normalized))

    def get_disparity_map(self):
        return self.disparity_Map
    
    def set_disparity_map(self, disparity_Map):
        self.disparity_Map = disparity_Map
    
    def get_stereo(self):
        return self.stereo

