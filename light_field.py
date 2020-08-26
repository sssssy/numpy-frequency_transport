import matplotlib.pyplot as plt
import numpy as np

from spectrum import *

class Ray(Spectrum):

    def __init__(self, p, d, s):
        super(Ray, self).__init__(dims = s.dims, radius = s.radius, 
            sampling_rate = s.sampling_rate, name = s.name)
        self.point = p
        self.direction = d

    def travel(self, d):
        new_matrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)))
        for i in range(self.sampling_rate):
            for j in range(self.sampling_rate):
                x = i-self.sampling_rate/2
                theta = j-self.sampling_rate/2
                new_matrix[i, j] = self.bilinear(self.matrix, x-d*theta, theta)
        del self.matrix
        del self.Fmatrix
        self.matrix = new_matrix
        self.fourier()

    def project(self, n):
        pass
        
    def reflect(self):
        pass

