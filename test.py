import numpy as np
import multiprocessing
from multiprocessing import Process
from multiprocessing import Pool
from time import perf_counter as clock

total = 100
cores = 4
l = np.zeros([total, total])

def loop(args):
    i, j = args
    l[i, j] = i*j

def a():
    for i in range(total*total // cores):
        processes = []
        for j in range(cores):
            current = i * cores + j
            ii = current // total
            jj = current % total
            if __name__ == '__main__':
                p = Process(target=loop, args=((ii, jj),))
                p.start()
                processes.append(p)
        if __name__ == '__main__':
            for p in processes:
                p.join()

def c():
    args = []
    for i in range(total):
        for j in range(total):
            args.append((i, j))
    with Pool() as p:
        p.map(loop, iter(args))
                
def b():
    for i in range(total):
        for j in range(total):
            l[i, j] = i+j

def time(f):
    t0 = clock()
    f()
    t1 = clock()
    print(t1 - t0)
    
if __name__ == '__main__':
    time(b)
    # time(c)
    time(a)