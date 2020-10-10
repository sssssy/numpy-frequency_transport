import cv2
import time
import sys

from spectrum import *
from spectrum4d import *
from ray import *
from ray4d import *

if __name__ == '__main__':
    # lena = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

    print('>>> 4D Light Field.')
    s = Spectrum4d(sampling_rate=20)
    p = np.array([0,0,0])
    d = np.array([0,0,1])
    r = Ray4d(p, d, s)
    # r.matrix = lena
    # r.set_cos()
    r.set_rect(2,1)
    r.visualize()

    r.travel(0.5)
    r.visualize()
    sys.exit()

    n = np.array([0., 1., -1.])
    n = n / np.linalg.norm(n)
    r.reparameter(n)
    r.visualize()

    r.curvature(0.5)
    r.visualize()
