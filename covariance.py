import numpy as np

from spectrum4d import *

class Covariance(Spectrum4d):

    def __init__(self, s):
        super(Covariance, self).__init__(dims = s.dims, radius = s.radius, 
            sampling_rate=s.sampling_rate, name=s.name, 
            fake_bilinear = s.fake_bilinear)
        self.m = np.identity(4)

    