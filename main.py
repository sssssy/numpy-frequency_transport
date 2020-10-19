# import cv2
import time
import sys
from time import perf_counter as clock

from ray4d import *
from covariance import *

if __name__ == '__main__':
    # lena = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

    s = Spectrum4d(sampling_rate=20,
            name='bsdf_cov_test',
            fake_bilinear=True)
    s = Covariance(s)
    p = np.array([0,0,0])
    d = np.array([0,0,1])
    r = Ray4d(p, d, s, bsdf_mode='ggx')
    print('>>> 4D Light Field. name: {}'.format(s.name))
    print('>>> fake_bilinear = {}\n'.format(s.fake_bilinear))

    t0 = clock()
    r.set_cos()
    # r.set_rect(4, 2)
    t1 = clock()
    print('\t>>  {}. {} \ttime: {:.4f}\n'.format(r.time, r.lastop, t1 - t0))
    r.visualize()

    r.visualize_bsdf()

#     t0 = clock()
#     r.travel(1)
#     t1 = clock()
#     print('\t>>  {}. {} \ttime: {:.4f}\n'.format(r.time, r.lastop, t1-t0))
#     r.visualize()

#     n = np.array([0., 1., -1.])
#     n = n / np.linalg.norm(n)
#     t0 = clock()
#     r.reparameter(n)
#     t1 = clock()
#     print('\t>>  {}. {} \ttime: {:.4f}\n'.format(r.time, r.lastop, t1-t0))
#     r.visualize()

#     t0 = clock()
#     r.curvature(0.5)
#     t1 = clock()
#     print('\t>>  {}. {} \ttime: {:.4f}\n'.format(r.time, r.lastop, t1-t0))
#     r.visualize()

#     t0 = clock()
#     r.mirror_reflection()
#     t1 = clock()
#     print('\t>>  {}. {} \ttime: {:.4f}\n'.format(r.time, r.lastop, t1-t0))
#     r.visualize()

#     t0 = clock()
#     r.cosine()
#     t1 = clock()
#     print('\t>>  {}. {} \ttime: {:.4f}\n'.format(r.time, r.lastop, t1-t0))
#     r.visualize()

    t0 = clock()
    r.bsdf_conv()
    t1 = clock()
    print('\t>>  {}. {} \ttime: {:.4f}\n'.format(r.time, r.lastop, t1-t0))
    r.visualize()