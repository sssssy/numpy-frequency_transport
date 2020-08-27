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
        self.lastop = 'travel'

        # apply primary domain operators
        new_matrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)))
        for i in range(self.sampling_rate):
            for j in range(self.sampling_rate):
                x = i-self.sampling_rate/2
                theta = j-self.sampling_rate/2
                new_matrix[i, j] = self.bilinear(self.matrix, x-d*theta, theta)

        # apply fourier domain operators
        del self.new_Fmatrix
        gc.collect()
        self.new_Fmatrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)), dtype=complex)
        for i in range(self.sampling_rate):
            for j in range(self.sampling_rate):
                x = i-self.sampling_rate/2
                theta = j-self.sampling_rate/2
                self.new_Fmatrix[i, j] = self.bilinear(self.Fmatrix, x, theta+d*x)

        # update matrices
        del self.matrix
        del self.Fmatrix
        gc.collect()
        self.matrix = new_matrix
        self.fourier()

    def reparameter(self, n):
        '''
        reparametrization in [Belcour]
        totally 3 steps, the 2nd step of primary domain is skipped in 2D condition
        '''
        self.time += 1
        self.lastop = 'reparameter'
        cos_alpha = -np.dot(self.direction, n)# / (|z|*|n|) == 1*1
        self.alpha = np.arccos(cos_alpha)

        # apply primary domain operators
        new_matrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)))
        for i in range(self.sampling_rate):
            for j in range(self.sampling_rate):
                x = i-self.sampling_rate/2
                theta = j-self.sampling_rate/2
                new_matrix[i, j] = self.bilinear(self.matrix, x*cos_alpha, 
                    self.alpha+theta*cos_alpha)

        # apply fourier domain operators
        del self.new_Fmatrix
        gc.collect()
        self.new_Fmatrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)), dtype=complex)
        for i in range(self.sampling_rate):
            for j in range(self.sampling_rate):
                x = i-self.sampling_rate/2
                theta = j-self.sampling_rate/2
                self.new_Fmatrix[i, j] = cos_alpha * self.bilinear(self.Fmatrix, x/cos_alpha, 
                    self.alpha+theta/cos_alpha)

        del self.matrix
        del self.Fmatrix
        gc.collect()
        self.matrix = new_matrix
        self.fourier()

    def curvature(self, K):
        self.time += 1
        self.lastop = 'curvature'

        # apply primary domain operators
        new_matrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)))
        for i in range(self.sampling_rate):
            for j in range(self.sampling_rate):
                x = i-self.sampling_rate/2
                theta = j-self.sampling_rate/2
                new_matrix[i, j] = self.bilinear(self.matrix, x, theta+K*(x+self.alpha))

        # apply fourier domain operators
        del self.new_Fmatrix
        gc.collect()
        self.new_Fmatrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)), dtype=complex)
        for i in range(self.sampling_rate):
            for j in range(self.sampling_rate):
                x = i-self.sampling_rate/2
                theta = j-self.sampling_rate/2
                self.new_Fmatrix[i, j] = np.exp(-2*1j*PI*K*self.alpha*theta)*\
                    self.bilinear(self.Fmatrix, x-K*theta, theta)

        del self.matrix
        del self.Fmatrix
        gc.collect()
        self.matrix = new_matrix
        self.fourier()

    def bsdf(self):
        pass
        
    def reflect(self):
        pass

    def visualize(self):
        fig = plt.figure()
        plt.suptitle(
            'Spectrum of "{}" after {}, time = {}\nSource: {}, Direction: {}'\
                .format(self.name, self.lastop, self.time, self.point, self.direction)
        )
        plt.subplots_adjust(wspace=0.5, hspace=0)

        ax0 = fig.add_subplot(131)
        ax0.imshow(np.transpose(self.matrix), origin='lower')
        plt.xlabel('x')
        plt.ylabel('theta (1/rad)')

        ax1 = fig.add_subplot(132)
        ax1.imshow(np.transpose(np.abs(self.Fmatrix)), origin='lower')
        plt.xlabel('w_X')
        plt.ylabel('w_THETA')

        if self.new_Fmatrix is not None:
            ax2 = fig.add_subplot(133)
            ax2.imshow(np.transpose(np.abs(self.new_Fmatrix)), origin='lower')
            plt.xlabel('w_X')
            plt.ylabel('w_THETA')

        # plt.show()
        plt.savefig('{}_{}'.format(self.name, self.time))

