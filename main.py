import numpy as np
import matplotlib.pyplot as plt
import time

from spectrum import *

if __name__ == '__main__':
    print('start.')

    s = Spectrum(name = 'cos')
    s.set_vcos()
    s.show()
