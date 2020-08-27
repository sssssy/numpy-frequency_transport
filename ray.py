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
        self.time += 1
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
        # TODO: apply fourier domain operator to validate

    def reparameter(self, n):
        '''
        reparametrization in [Belcour]
        total 3 steps, the 2nd step is skipped in 2D condition
        '''
        self.time += 1
        cos_alpha = -np.dot(self.direction, n)# / (|z|*|n|) == 1*1
        alpha = np.arccos(cos_alpha)
        new_matrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)))
        for i in range(self.sampling_rate):
            for j in range(self.sampling_rate):
                x = i-self.sampling_rate/2
                theta = j-self.sampling_rate/2
                new_matrix[i, j] = self.bilinear(self.matrix, x*cos_alpha, alpha+theta)
        del self.matrix
        del self.Fmatrix
        self.matrix = new_matrix
        self.fourier()
        # TODO: apply fourier domain operator to validate
        
    def reflect(self):
        pass

