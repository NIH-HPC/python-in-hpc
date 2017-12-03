#! /usr/bin/env python
"""
calculates a mandel set using block distribution - i.e.
    rank 0 calulates lines [0,n1[
    rank 1                 [n1,n2[

Run with
    mpiexec [mpiexec otions] ./mandelbrot_mpi.py

This code was developed with
   mpi4py                    2.0.0 
   numba                     0.35.0
   numpy                     1.11.3
   python                    2.7.13
"""

###
### imports
###

from mpi4py import MPI
import numpy as np
from numba import jit

tic = MPI.Wtime()

###
### globals
###

# area1:
#xmin, xmax = -2.0, 0.5
#ymin, ymax =  -1.25,  1.25
# maxiter = 80

# area2:
xmin, xmax = -0.74877, -0.74872
ymin, ymax =  0.06505,  0.06510
width, height = 3000, 3000
maxiter = 2048

dy = (ymax - ymin) / (height - 1)

###
### functions
###

@jit
def mandel(creal, cimag, maxiter):
    real = creal
    imag = cimag
    for n in range(maxiter):
        real2 = real*real
        imag2 = imag*imag
        if real2 + imag2 > 4.0:
            return n
        imag = 2 * real*imag + cimag
        real = real2 - imag2 + creal
    return n

@jit
def mandel_set(xmin, xmax, ymin, ymax, width, height, maxiter):
    r = np.linspace(xmin, xmax, width)
    i = np.linspace(ymin, ymax, height)
    Cm = np.empty((height, width), dtype='i')
    for x in range(width):
        for y in range(height):
            Cm[y, x] = mandel(r[x], i[y], maxiter)
    return Cm 

###
### main
###

comm  = MPI.COMM_WORLD
size  = comm.Get_size()
rank  = comm.Get_rank()

print "Rank {:4d}: checking in".format(rank)

# how many rows to compute in this rank?
#   for example: height 100
#                size 8
#   then height % size = 4
#   which means that ranks 0 - 3 each do one
#   more row (13) than ranks 4 - 7 (12)
N = height // size + (height % size > rank)
print "Rank {}: will compute {} rows".format(rank, N)
N = np.array(N, dtype='i')  # so we can Gather it later on

# first row to compute here
#   scan: the operation returns for each rank i the sum of send buffers of ranks [0,i]
# start_y and end_y are the first and last y value to calculate in this block
start_i = comm.scan(N) - N
start_y = ymin + start_i * dy
end_y   = ymin + (start_i + N - 1) * dy
print "Rank {:4d}: will compute y = [{}, {}]".format(rank, start_y, end_y)

# calculate the local results
Cl = mandel_set(xmin, xmax, start_y, end_y, width, N, maxiter)
print "Rank {:4d}: finished computing rows; result matrix is shape {}".format(rank, Cl.shape)
print "Rank {:4d}: max value in array: {}".format(rank, Cl.max())

# gather the number of rows calculated by each rank. Note that the N of each rank
# was wrapped in a numpy array so here the upper case 'Gather' can be used. This
# is faster than using the lower case 'gather' which is meant for python objects.
# Though this is tiny data so it would not have mattered.
rowcounts = 0     # has to be zero, not None b/c of the 'rowcounts * width' bit later on
C         = None
if rank == 0:
    rowcounts = np.empty(size, dtype='i')
    C = np.zeros([height, width], dtype='i')

comm.Gather(sendbuf = [N, MPI.INT], 
            recvbuf = [rowcounts, MPI.INT], 
            root    = 0)

# gather the global results matrix
#  note: Gatherv allows varying amounts of data from each rank. In the underlying
#        MPI implementation the receiving buffer has to specify how many elements to
#        expect from each rank, and at what position they should be inserved into the
#        receiver buffer. I think the 'None' make mpi4py automatically figure out
#        the displacements. There is very little documentation on Gatherv in mpi4py and
#        the examples i've found all differ.

comm.Gatherv(sendbuf = [Cl, MPI.INT], 
             recvbuf = [C, (rowcounts * width, None), MPI.INT], 
             root    = 0)

toc = MPI.Wtime()

wct = comm.gather(toc - tic, root=0)
if rank == 0:
    for task, time in enumerate(wct):
        print "Rank {:4d}: ran for {:8.2f}s".format(task, time)
    print "max(runtime)  = {:8.2f}s".format(max(wct))
    print "min(runtime ) = {:8.2f}s".format(min(wct))
    print "mean(runtime) = {:8.2f}s".format(sum(wct) / len(wct))
    print "Array size: {} x {}".format(height, width)

# eye candy (requires matplotlib)
if rank == 0 and width * height <= 1e7:
    try:
        from matplotlib import pyplot as plt
        from matplotlib import colors
    except ImportError:
        print ('No matplotlib found; skipping plot')
    else:
        norm = colors.PowerNorm(0.3)
        figsz = max(width, height) / 100
        fig = plt.figure(figsize=(figsz, figsz), dpi=100, tight_layout=True)
        ax = fig.add_subplot(111)
        ax.imshow(C, cmap='magma', norm=norm, origin='lower', aspect='equal')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig('mandelbrot.png')
MPI.COMM_WORLD.Barrier()
