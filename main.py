import cv2
import time

from spectrum import *
from ray import *

if __name__ == '__main__':
    print('start.')
    # lena = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

    s = Spectrum(sampling_rate=100, name='cos')
    p = np.array([0,0])
    d = np.array([1,0])
    r = Ray(p, d, s)
    # r.matrix = lena
    # r.set_cos()
    r.set_rect(2,1)
    r.travel(1)
    r.show()
    n = np.array([-1., 0.])
    n = n / np.linalg.norm(n)
    r.reparameter(n)
    r.show()
