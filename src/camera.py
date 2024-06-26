import numpy as np # type: ignore

class Calibration:
    def __init__(self, calibFile):
        self.calibFile = calibFile
        self.cam0 = None
        self.cam1 = None
        self.doffs = None
        self.baseline = None
        self.width = None
        self.height = None
        self.ndisp = None
        self.vmin = None
        self.vmax = None
        self.CalibrateDepthMap()

    def CalibrateDepthMap(self):

        with open(self.calibFile, 'r') as file:
            data = file.read()

        # Process each line
        for line in data.split('\n'):
            if 'cam0' in line:
                cam0_str = line.split('=')[1].strip()[1:-1]
                self.cam0 = np.array([list(map(float, row.split())) for row in cam0_str.split(';')])
            elif 'cam1' in line:
                cam1_str = line.split('=')[1].strip()[1:-1]
                self.cam1 = np.array([list(map(float, row.split())) for row in cam1_str.split(';')])
            elif 'doffs' in line:
                self.doffs = float(line.split('=')[1].strip())
            elif 'baseline' in line:
                self.baseline = float(line.split('=')[1].strip())
            elif 'width' in line:
                self.width = int(line.split('=')[1].strip())
            elif 'height' in line:
                self.height = int(line.split('=')[1].strip())
            elif 'ndisp' in line:
                self.ndisp = int(line.split('=')[1].strip())
            elif 'vmin' in line:
                self.vmin = int(line.split('=')[1].strip())
            elif 'vmax' in line:
                self.vmax = int(line.split('=')[1].strip())

    def get_baseline(self):
        return self.baseline
    
    def get_cam0(self):
        return self.cam0
    
    def get_cam1(self):
        return self.cam1

    def get_doffs(self):
        return self.doffs
    
    def get_baseline(self):
        return self.baseline
    
    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def get_ndisp(self):
        return self.ndisp
    
    def get_vmin(self):
        return self.vmin
    
    def get_vmax(self):
        return self.vmax

    # note to self create getters for all variables