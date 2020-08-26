import cv2
import time

from spectrum import *
from light_field import *

if __name__ == '__main__':
    print('start.')
    # lena = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

    s = Spectrum(name='cos')
    r = Ray(0, 0, s) # TODO: point, direction
    # r.matrix = lena
    # r.set_cos()
    r.set_rect(1,1)
    r.travel(0.2)
    r.show()
