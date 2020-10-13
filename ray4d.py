import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

from spectrum4d import *

class Ray4d(Spectrum4d):

    def __init__(self, p, d, s):
        super(Ray4d, self).__init__(dims = s.dims, radius = s.radius, 
            sampling_rate=s.sampling_rate, name=s.name, 
            fake_bilinear = s.fake_bilinear)
        self.point = p
        self.direction = d
        self.theta_in = 0.5
        self.theta_out = 1.0
        self.bsdf = self.init_bsdf(mode = 'white')

    def init_bsdf(self, mode):
        
        if mode == 'white':
            bsdf = np.random.normal(0, 1,
                [self.sampling_rate, self.sampling_rate])
            bsdf = bsdf / np.max(np.abs(bsdf)) / 2 + 0.5
            return bsdf

        if mode == 'cos':
            pass

        if mode == 'gaussian':
            pass

    def travel(self, d):
        '''
        4d condition
        '''
        self.time += 1
        self.lastop = 'travel_4d'

        # apply primary domain operators
        # new_matrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)))
        new_matrix = deepcopy(self.matrix)
        for i in range(self.sampling_rate):
            for j in range(self.sampling_rate):
                for k in range(self.sampling_rate):
                    for l in range(self.sampling_rate):

                        x = i-self.sampling_rate/2
                        y = j-self.sampling_rate/2
                        theta = k-self.sampling_rate/2
                        phi = l - self.sampling_rate / 2

                        i_ = round(x - d * theta) + self.sampling_rate//2
                        j_ = round(y - d * phi) + self.sampling_rate//2
                        k_ = round(theta) + self.sampling_rate//2
                        l_ = round(phi) + self.sampling_rate//2

                        if not all([_ < self.sampling_rate and _ >= 0
                            for _ in (i_, j_, k_, l_)
                            ]
                        ): # TODO: make more accurate
                            new_matrix[i, j, k, l] = UNDEFINED
                        
                        else:
                            new_matrix[i, j, k, l] = self.matrix[
                                i_,
                                j_,
                                k_,
                                l_
                                ]


        # apply fourier domain operators
        # del self.new_Fmatrix
        # gc.collect()
        # self.new_Fmatrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)), dtype=complex)
        # for i in range(self.sampling_rate):
        #     for j in range(self.sampling_rate):
        #         for k in range(self.sampling_rate):
        #             for l in range(self.sampling_rate):
        #                 x = i-self.sampling_rate/2
        #                 y = j-self.sampling_rate/2
        #                 theta = k-self.sampling_rate/2
        #                 phi = l-self.sampling_rate/2
        #                 self.new_Fmatrix[i, j, k, l] = self.bilinear(self.Fmatrix,
        #                     x, 
        #                     y,
        #                     theta+d*x,
        #                     phi+d*y)
        #                 # self.new_Fmatrix[i, j, k, l] = 0

        # update matrices
        del self.matrix
        # del self.Fmatrix
        # gc.collect()
        self.matrix = new_matrix
        self.fourier()

    def reparameter(self, n):
        '''
        4d condition
        '''
        self.time += 1
        self.lastop = 'reparameter_4d'
        cos_alpha = -np.dot(self.direction, n)  # / (|z|*|n|) == 1*1
        sin_alpha = np.sqrt(1-cos_alpha**2)
        self.alpha = np.arccos(cos_alpha)
        # print(cos_alpha)

        # apply primary domain operators
        new_matrix = deepcopy(self.matrix)
        # new_matrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)))
        for i in range(self.sampling_rate):
            for j in range(self.sampling_rate):
                for k in range(self.sampling_rate):
                    for l in range(self.sampling_rate):

                        x = i-self.sampling_rate/2
                        y = j-self.sampling_rate/2
                        theta = k-self.sampling_rate/2
                        phi = l - self.sampling_rate / 2
                        
                        i_ = round(x * cos_alpha - y * sin_alpha) + self.sampling_rate//2
                        j_ = round(x * sin_alpha / cos_alpha + y) + self.sampling_rate//2
                        k_ = round(theta * cos_alpha - phi * sin_alpha) + self.sampling_rate//2
                        l_ = round(theta * sin_alpha + phi * cos_alpha) + self.sampling_rate//2

                        if not all([_ < self.sampling_rate and _ >= 0
                            for _ in (i_, j_, k_, l_)
                            ]
                        ): # TODO: make more accurate
                            new_matrix[i, j, k, l] = UNDEFINED
                        
                        else:
                            new_matrix[i, j, k, l] = self.matrix[
                                int(i_),
                                int(j_),
                                int(k_),
                                int(l_)
                            ]
                        # new_matrix[i, j, k, l] = 0

        # # apply fourier domain operators
        # del self.new_Fmatrix
        # gc.collect()
        # self.new_Fmatrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)), dtype=complex)
        # for i in range(self.sampling_rate):
        #     for j in range(self.sampling_rate):
        #         x = i-self.sampling_rate/2
        #         theta = j-self.sampling_rate/2
        #         self.new_Fmatrix[i, j] = cos_alpha * self.bilinear(self.Fmatrix, x/cos_alpha, 
        #             self.alpha+theta/cos_alpha)

        del self.matrix
        # del self.Fmatrix
        # gc.collect()
        self.matrix = new_matrix
        self.fourier()

    def curvature(self, K):
        '''
        4d condition
        '''
        self.time += 1
        self.lastop = 'curvature_4d'

        # apply primary domain operators
        new_matrix = deepcopy(self.matrix)
        # new_matrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)))
        for i in range(self.sampling_rate):
            for j in range(self.sampling_rate):
                for k in range(self.sampling_rate):
                    for l in range(self.sampling_rate):
                        x = i-self.sampling_rate/2
                        y = j-self.sampling_rate/2
                        theta = k-self.sampling_rate/2
                        phi = l-self.sampling_rate/2

                        i_ = round(x) + self.sampling_rate//2
                        j_ = round(y) + self.sampling_rate//2
                        k_ = round(theta + K * (x + self.alpha)) + self.sampling_rate//2
                        l_ = round(phi + K * y) + self.sampling_rate//2

                        if not all([_ < self.sampling_rate and _ >= 0
                            for _ in (i_, j_, k_, l_)
                            ]
                        ): # TODO: make more accurate
                            new_matrix[i, j, k, l] = UNDEFINED
                        
                        else:
                            new_matrix[i, j, k, l] = self.matrix[
                                int(i_),
                                int(j_),
                                int(k_),
                                int(l_)
                            ]

        # # apply fourier domain operators
        # del self.new_Fmatrix
        # gc.collect()
        # self.new_Fmatrix = np.zeros(tuple(self.sampling_rate for i in range(self.dims)), dtype=complex)
        # for i in range(self.sampling_rate):
        #     for j in range(self.sampling_rate):
        #         x = i-self.sampling_rate/2
        #         theta = j-self.sampling_rate/2
        #         self.new_Fmatrix[i, j] = np.exp(-2*1j*PI*K*self.alpha*theta)*\
        #             self.bilinear(self.Fmatrix, x-K*theta, theta)

        del self.matrix
        # del self.Fmatrix
        # gc.collect()
        self.matrix = new_matrix
        self.fourier()

    def mirror_reflection(self):
        
        self.time += 1
        self.lastop = 'mirror_reflection_4d'

        # apply primary domain operators
        new_matrix = deepcopy(self.matrix)
        for i in range(self.sampling_rate):
            for j in range(self.sampling_rate):
                for k in range(self.sampling_rate):
                    for l in range(self.sampling_rate):
                        x = i-self.sampling_rate/2
                        y = j-self.sampling_rate/2
                        theta = k-self.sampling_rate/2
                        phi = l-self.sampling_rate/2

                        i_ = round(x) + self.sampling_rate//2
                        j_ = round(y) + self.sampling_rate//2
                        k_ = round(-theta) + self.sampling_rate//2
                        l_ = round(-phi) + self.sampling_rate//2

                        if not all([_ < self.sampling_rate and _ >= 0
                            for _ in (i_, j_, k_, l_)
                            ]
                        ): # TODO: make more accurate
                            new_matrix[i, j, k, l] = UNDEFINED
                        
                        else:
                            new_matrix[i, j, k, l] = self.matrix[
                                int(i_),
                                int(j_),
                                int(k_),
                                int(l_)
                            ]

        del self.matrix
        self.matrix = new_matrix
        self.fourier()

    def cosine(self):

        self.time += 1
        self.lastop = 'cosine_4d'

        # apply primary domain operators
        new_matrix = deepcopy(self.matrix)
        for i in range(self.sampling_rate):
            for j in range(self.sampling_rate):
                for k in range(self.sampling_rate):
                    for l in range(self.sampling_rate):
                        x = i-self.sampling_rate/2
                        y = j-self.sampling_rate/2
                        theta = k-self.sampling_rate/2
                        phi = l-self.sampling_rate/2

                        i_ = round(x * np.cos(theta + self.theta_in)) + self.sampling_rate//2
                        j_ = round(y * np.cos(phi + self.theta_in)) + self.sampling_rate//2
                        k_ = round(theta * np.cos(theta + self.theta_in)) + self.sampling_rate//2
                        l_ = round(phi * np.cos(phi + self.theta_in)) + self.sampling_rate//2

                        if not all([_ < self.sampling_rate and _ >= 0
                            for _ in (i_, j_, k_, l_)
                            ]
                        ): # TODO: make more accurate
                            new_matrix[i, j, k, l] = UNDEFINED
                        
                        else:
                            new_matrix[i, j, k, l] = self.matrix[
                                int(i_),
                                int(j_),
                                int(k_),
                                int(l_)
                        ]

        del self.matrix
        self.matrix = new_matrix
        self.fourier()

    def bsdf_conv(self):

        self.time += 1
        self.lastop = 'bsdf_conv_4d'

        # apply primary domain operators
        new_matrix = deepcopy(self.matrix)
        for i in range(self.sampling_rate):
            for j in range(self.sampling_rate):
                for k in range(self.sampling_rate):
                    for l in range(self.sampling_rate):
                        x = i-self.sampling_rate//2
                        y = j-self.sampling_rate//2
                        theta = k-self.sampling_rate//2
                        phi = l-self.sampling_rate//2
                        new_matrix[i, j, k, l] = self.integrate_bsdf(
                            i,
                            j,
                            k,
                            l
                        )

        del self.matrix
        self.matrix = new_matrix
        self.fourier()

    def integrate_bsdf(self, i, j, k, l):
        theta_sum = 0
        for theta_in in range(self.sampling_rate):
            theta_sum += self.matrix[i, j, theta_in, l] * self.bsdf[theta_in, k]
        return theta_sum

    def visualize(self):
        '''
        4d condition
        '''
        fig = plt.figure()
        plt.suptitle(
            'Spectrum of "{}(4d)" after {}, time = {}\n' #Source: {}, Direction: {}'\
                .format(self.name, self.lastop, self.time, self.point, self.direction)
        )
        plt.subplots_adjust(wspace=0.5, hspace=0)

        ax0 = fig.add_subplot(121)
        ax0.imshow(np.transpose(self.matrix[:,
            self.sampling_rate//2,
            :,
            self.sampling_rate//2]), origin='lower')
        plt.xlabel('x')
        plt.ylabel('theta (1/rad)')

        ax1 = fig.add_subplot(122)
        ax1.imshow(np.transpose(np.abs(self.Fmatrix[:,
            self.sampling_rate//2,
            :,
            self.sampling_rate//2])), origin='lower')
        plt.xlabel('w_X')
        plt.ylabel('w_THETA')

        # if self.new_Fmatrix is not None:
        #     ax2 = fig.add_subplot(133)
        #     ax2.imshow(np.transpose(np.abs(self.new_Fmatrix[:,
        #         self.sampling_rate//2,
        #         :,
        #         self.sampling_rate//2])), origin='lower')
        #     plt.xlabel('w_X')
        #     plt.ylabel('w_THETA')

        # plt.show()
        plt.savefig('images/{}d_{}_{}'.format(self.dims, self.name, self.time))