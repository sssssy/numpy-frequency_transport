import numpy as np
from pylab import *
import matplotlib.pyplot as plt

PI = 3.1415926

class Spectrum():

    def __init__(self, dims=2, radius=PI, sampling_rate=1000, name='unnamed spectrum'):
        self.time = 0
        self.name = name
        self.dims = dims
        self.radius = radius
        self.sampling_rate = sampling_rate
        self.width = 2 * self.radius / self.sampling_rate
        self.matrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)))
        self.Fmatrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)))

    def set_vcos(self):
        cos_list = np.cos(np.arange(-self.radius, self.radius, 2*self.radius/1000.))
        cos_list = np.expand_dims(cos_list, 0)
        cos_matrix = np.repeat(cos_list, 1000, axis=0)
        self.matrix = cos_matrix
    
    def set_hcos(self):
        cos_list = np.cos(np.arange(-self.radius, self.radius, 2*self.radius/1000.))
        cos_list = np.expand_dims(cos_list, 1)
        cos_matrix = np.repeat(cos_list, 1000, axis=1)
        self.matrix = cos_matrix

    def show(self):
        fig = plt.figure()
        plt.subplots_adjust(wspace=0.5, hspace=0)

        ax0 = fig.add_subplot(121)
        ax0.imshow(self.matrix)
        plt.title('spectrum of '+self.name+": "+str(self.time))
        plt.xlabel('theta (1/rad)')
        plt.ylabel('x')

        ax1 = fig.add_subplot(122)
        ax1 = imshow(self.Fmatrix)
        plt.title('fourier domain of '+self.name+": "+str(self.time))
        plt.xlabel('w_THETA')
        plt.ylabel('w_X')

        plt.show()

