import cv2
import time

from spectrum import *
from ray import *

if __name__ == '__main__':
    # lena = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

    s = Spectrum(sampling_rate=100)
    p = np.array([0,0,0])
    d = np.array([0,0,1])
    r = Ray(p, d, s)
    # r.matrix = lena
    # r.set_cos()
    r.set_rect(2,1)
    r.visualize()

    r.travel(0.5)
    r.visualize()

    n = np.array([0., 1., -1.])
    n = n / np.linalg.norm(n)
    r.reparameter(n)
    r.visualize()

    r.curvature(0.5)
    r.visualize()
