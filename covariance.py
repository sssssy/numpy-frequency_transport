import numpy as np

from spectrum4d import *

class Covariance(Spectrum4d):

    def __init__(self, s):

        super(Covariance, self).__init__(dims = s.dims, radius = s.radius, 
            sampling_rate=s.sampling_rate, name=s.name, 
            fake_bilinear = s.fake_bilinear)
        self.cov = None

    def set_eye(self):

        self.cov = np.identity(self.dims)

    def get_cov(self, m):

        # calculate cov of self.matrix
        pass

    def apply_transform(self):

        self.cov = np.matmul(self.m.T, self.cov)
        self.cov = np.matmul(self.cov, self.m)

    def travel_cov(self, d):
        
        self.m = np.identity(4)
        self.m[0, 2] = self.m[1, 3] = -d
        self.apply_transform()

    def rotation_cov(self, alpha):

        self.m = np.zeros(self.cov.shape)
        self.m[0, 0] = self.m[1, 1] = self.m[2, 2] = self.m[3, 3] = np.cos(alpha)
        self.m[0, 1] = self.m[2, 3] = -np.sin(alpha)
        self.m[1, 0] = self.m[3, 2] = np.sin(alpha)
        self.apply_transform()

    def scale_cov(self, alpha):

        self.m = np.identity(self.dims)
        self.m[1, 1] = alpha
        self.apply_transform()
    
    def curvature_cov(self, K):

        self.m = identity(self.dims)
        self.m[2, 0] = self.m[3, 1] = -K
        self.apply_transform()

    def mirror_reflection_cov(self):

        self.m = identity(self.dims)
        self.m[2, 2] = self.m[3, 3] = -1
        self.apply_transform()

    def cosine_cov(self, theta_in):
        
        # calculate cov of cosine light field
        cos_light_field = np.zeros(self.matrix.shape)
        for i in range(self.sampling_rate):
            for j in range(self.sampling_rate):
                for k in range(self.sampling_rate):
                    for l in range(self.sampling_rate):
                        cos_light_field[i, j, k, l] = max(0, np.cos(k - self.sampling_rate // 2 + theta_in))
        cos_light_field_cov = self.get_cov(cos_light_field)

        # add it to self.cov
        self.cov += cos_light_field_cov

    def print_cov(self):

        print(self.cov)