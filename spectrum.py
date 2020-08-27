import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import gc

PI = 3.1415926
UNDEFINED = 0

class Spectrum():
    '''
    matrix[x, theta]
    '''

    def __init__(self, dims=2, radius=10, sampling_rate=100, name='unnamed spectrum'):
        assert (dims==2), "only 2-d examples is supported yet."
        self.time = 0
        self.name = name
        self.dims = dims
        self.radius = radius
        self.sampling_rate = sampling_rate
        self.width = 2 * self.radius / self.sampling_rate
        self.init_matrix()
        self.init_Fmatrix()
        self.lastop = 'init'

    def init_matrix(self):
        self.matrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)))

    def init_Fmatrix(self):
        self.Fmatrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)))

    def set_cos(self):
        self.lastop = 'set_cos'
        cos_list = np.cos(np.arange(-self.radius, self.radius, 2*self.radius/self.sampling_rate))
        normalized_cos_list = 255*(0.5+cos_list/2)
        normalized_cos_list = normalized_cos_list.astype(np.uint8)
        normalized_cos_list = np.expand_dims(normalized_cos_list, 0)
        cos_matrix = np.repeat(normalized_cos_list, self.sampling_rate, axis=0)
        del self.matrix
        del self.Fmatrix
        gc.collect()
        self.matrix = cos_matrix
        self.fourier()

    def set_rect(self, width, height, value=255):
        self.lastop = 'set_rect'
        w_sampling_radius = int(width/self.radius*self.sampling_rate/2)
        h_sampling_radius = int(height/self.radius*self.sampling_rate/2)
        w_sampling_start = int(self.sampling_rate/2 - w_sampling_radius)
        h_sampling_start = int(self.sampling_rate/2 - h_sampling_radius)
        del self.matrix
        del self.Fmatrix
        gc.collect()
        self.init_matrix()
        self.matrix[w_sampling_start:w_sampling_start+w_sampling_radius*2,\
            h_sampling_start:h_sampling_start+h_sampling_radius*2] = value
        self.fourier()
    
    def fourier(self):
        self.Fmatrix = np.fft.fftshift(np.fft.fft2(self.matrix))

    def bilinear(self, m, i, j):
        i = i+self.sampling_rate/2
        j = j+self.sampling_rate/2
        i_floor = int(np.floor(i))
        i_ceil = int(np.ceil(i))
        j_floor = int(np.floor(j))
        j_ceil = int(np.ceil(j))
        if not all([x < self.sampling_rate and x >= 0 
            for x in (i_floor, i_ceil, j_floor, j_ceil)]): # TODO: make more accurate
            return UNDEFINED
        res_floor = np.interp(i, [i_floor, i_ceil], [m[i_floor, j_floor], m[i_ceil, j_floor]])
        res_ceil = np.interp(i, [i_floor, i_ceil], [m[i_floor, j_ceil], m[i_ceil, j_ceil]])
        res = np.interp(j, [j_floor, j_ceil], [res_floor, res_ceil])

        return res
