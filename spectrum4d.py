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

    def __init__(self, dims=4, radius=10, sampling_rate=20,
            fake_bilinear=False,name='unnamed'):
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
        self.fake_bilinear = fake_bilinear

    def init_matrix(self):
        self.matrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)))

    def init_Fmatrix(self):
        self.Fmatrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)), dtype=complex)

    def set_cos(self):
        '''
        [x, y, theta, phi] = [~, ~, cos, ~]
        ~ represents a uniform distribution, 
        i.e. same in every channel of this dimension.
        '''
        self.lastop = 'set_cos_4d'
        
        coord = np.linspace(0, PI * 2, self.sampling_rate)
        dist = -np.cos(coord)
        dist = np.expand_dims(dist, axis=(0, 1, 3))
        dist = np.repeat(dist, self.sampling_rate, 0)
        dist = np.repeat(dist, self.sampling_rate, 1)
        dist = np.repeat(dist, self.sampling_rate, 3)

        del self.matrix
        del self.Fmatrix
        self.init_matrix()

        self.matrix = dist
        self.fourier()

        self.get_cov(self.matrix)
    
    def set_rect(self, width, height, value=255):
        '''
        4d condition
        [x, y, theta, phi] = [rect, ~, rect, ~]
        only set x, theta dims,
        repeat the same in y & phi dims.
        '''
        self.lastop = 'set_rect4d'
        # self.name += '_rect4d'
        w_sampling_radius = int(width/self.radius*self.sampling_rate/2)
        h_sampling_radius = int(height/self.radius*self.sampling_rate/2)
        w_sampling_start = int(self.sampling_rate/2 - w_sampling_radius)
        h_sampling_start = int(self.sampling_rate / 2 - h_sampling_radius)
        
        del self.matrix
        del self.Fmatrix
        # gc.collect()
        self.init_matrix()
        self.matrix[w_sampling_start:w_sampling_start+w_sampling_radius*2,
            :,
            h_sampling_start:h_sampling_start+h_sampling_radius*2,
            :] = value
        self.fourier()

        self.get_cov(self.matrix)

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
        if self.fake_bilinear:
            res = m[i_floor, j_floor, k_ceil, l_ceil]
        else:
            res = interpn(coords, m, [i, j, k, l])
        return res
