# import cv2
import time
import sys
from time import perf_counter as clock

# from spectrum import *
from spectrum4d import *
# from ray import *
from ray4d import *

if __name__ == '__main__':
    # lena = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

    s = Spectrum4d(sampling_rate=40, fake_bilinear=True)
    print('>>> 4D Light Field.')
    print('>>> fake_bilinear = {}'.format(s.fake_bilinear))
    p = np.array([0,0,0])
    d = np.array([0,0,1])
    r = Ray4d(p, d, s)
    # r.matrix = lena
    # r.set_cos()
    t0 = clock()
    r.set_rect(2, 1)
    t1 = clock()
    print('\t{} time: {}'.format(r.lastop, t1-t0))
    r.visualize()

    t0 = clock()
    r.travel(1)
    t1 = clock()
    print('\t{} time: {}'.format(r.lastop, t1-t0))
    r.visualize()
    sys.exit()

    n = np.array([0., 1., -1.])
    n = n / np.linalg.norm(n)
    r.reparameter(n)
    r.visualize()

    r.curvature(0.5)
    r.visualize()
