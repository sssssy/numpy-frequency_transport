import numpy as np
from scipy.interpolate import interpn
from pylab import *
import matplotlib.pyplot as plt
import gc

PI = 3.1415926
UNDEFINED = 0

class Spectrum4d():
    '''
    matrix[x, theta]
    '''

    def __init__(self, dims=4, radius=10, sampling_rate=20, name='unnamed'):
        assert(dims==4), 'error, this is a 4d configuration'
        self.time = 0
        self.name = name
        self.dims = dims
        self.radius = radius
        self.sampling_rate = sampling_rate
        self.width = 2 * self.radius / self.sampling_rate
        self.init_matrix()
        self.init_Fmatrix()
        self.new_Fmatrix = None
        self.lastop = 'init'

    def init_matrix(self):
        self.matrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)))

    def init_Fmatrix(self):
        self.Fmatrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)), dtype=complex)

    def set_cos(self):
        pass
    
    def set_rect(self, width, height, value=255):
        '''
        4d condition
        only set x, theta dims,
        repeat the same in y & phi dims.
        '''
        self.lastop = 'set_rect4d'
        self.name += '_rect4d'
        w_sampling_radius = int(width/self.radius*self.sampling_rate/2)
        h_sampling_radius = int(height/self.radius*self.sampling_rate/2)
        w_sampling_start = int(self.sampling_rate/2 - w_sampling_radius)
        h_sampling_start = int(self.sampling_rate/2 - h_sampling_radius)
        del self.matrix
        del self.Fmatrix
        gc.collect()
        self.init_matrix()
        self.matrix[w_sampling_start:w_sampling_start+w_sampling_radius*2,
            :,
            h_sampling_start:h_sampling_start+h_sampling_radius*2,
            :] = value
        self.fourier()

    def fourier(self):
        self.Fmatrix = np.fft.fftshift(np.fft.fftn(self.matrix, axes=(0,1,2,3)))

    def bilinear(self, m, i, j, k, l):
        '''
        4d condition
        '''
        i = i+self.sampling_rate/2
        j = j+self.sampling_rate/2
        k = k+self.sampling_rate/2
        l = l+self.sampling_rate/2
        i_floor = int(np.floor(i))
        i_ceil = int(np.ceil(i))
        j_floor = int(np.floor(j))
        j_ceil = int(np.ceil(j))
        k_floor = int(np.floor(k))
        k_ceil = int(np.ceil(k))
        l_floor = int(np.floor(l))
        l_ceil = int(np.ceil(l))

        if not all([x < self.sampling_rate and x >= 0 
            for x in (i_floor, i_ceil, 
                j_floor, j_ceil,
                k_floor, k_ceil,
                l_floor, l_ceil)
            ]
        ): # TODO: make more accurate
            return UNDEFINED

        coords = tuple(np.arange(self.sampling_rate) for i in range(self.dims))
        res = interpn(coords, m, [i, j, k, l])
        return res
