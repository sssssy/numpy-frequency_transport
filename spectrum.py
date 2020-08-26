import numpy as np
from pylab import *
import matplotlib.pyplot as plt

PI = 3.1415926

class Spectrum():

    def __init__(self, dims=2, radius=10, sampling_rate=1000, name='unnamed spectrum'):
        assert (dims==2), "only 2-d examples is supported yet."
        self.time = 0
        self.name = name
        self.dims = dims
        self.radius = radius
        self.sampling_rate = sampling_rate
        self.width = 2 * self.radius / self.sampling_rate
        self.reset_matrix()
        self.reset_Fmatrix()

    def reset_matrix(self):
        self.matrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)))

    def reset_Fmatrix(self):
        self.Fmatrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)))

    def set_cos(self):
        cos_list = np.cos(np.arange(-self.radius, self.radius, 2*self.radius/self.sampling_rate))
        normalized_cos_list = 255*(0.5+cos_list/2)
        normalized_cos_list = normalized_cos_list.astype(np.uint8)
        normalized_cos_list = np.expand_dims(normalized_cos_list, 0)
        cos_matrix = np.repeat(normalized_cos_list, 1000, axis=0)
        self.matrix = cos_matrix

    def set_rect(self, width, height, value=255):
        w_sampling_radius = int(width/self.radius*self.sampling_rate/2)
        h_sampling_radius = int(height/self.radius*self.sampling_rate/2)
        w_sampling_start = int(self.sampling_rate/2 - w_sampling_radius)
        h_sampling_start = int(self.sampling_rate/2 - h_sampling_radius)
        self.reset_matrix()
        self.matrix[h_sampling_start:h_sampling_start+h_sampling_radius*2,\
            w_sampling_start:w_sampling_start+w_sampling_radius*2] = value
    
    def fourier(self):
        self.Fmatrix = np.fft.fftshift(np.fft.fft2(self.matrix))

    def show(self):
        fig = plt.figure()
        plt.subplots_adjust(wspace=0.5, hspace=0)

        ax0 = fig.add_subplot(121)
        ax0.imshow(self.matrix)
        plt.title('spectrum of '+self.name+": "+str(self.time))
        plt.xlabel('theta (1/rad)')
        plt.ylabel('x')

        ax1 = fig.add_subplot(122)
        ax1.imshow(np.abs(self.Fmatrix))
        plt.title('fourier domain of '+self.name+": "+str(self.time))
        plt.xlabel('w_THETA')
        plt.ylabel('w_X')

        plt.show()

