import cv2
import time

from spectrum import *

if __name__ == '__main__':
    print('start.')
    # lena = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

    s = Spectrum(name = 'cos')
    # s.matrix = lena
    # s.set_cos()
    s.set_rect(1,1)
    s.fourier()
    s.show()
