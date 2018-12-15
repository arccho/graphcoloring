from GraphColor import *
import pycuda.driver as cuda
import pycuda.compiler
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import pycuda.curandom
from pycuda import characterize
from pycuda.compiler import SourceModule
import numpy as np

MyGraph = GraphColor("net50k001.txt")

# TODO: implementer methode MCMC

g = pycuda.curandom.XORWOWRandomNumberGenerator()
size = np.array([MyGraph.graphStruct.node])
cuda.mem_alloc(size*characterize.sizeof('curandStateXORWOW', '#include <curand_kernel.h>'))