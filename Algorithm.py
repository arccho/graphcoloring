from GraphColor import *
import pycuda.driver as cuda
import pycuda.compiler
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import pycuda.curandom
from pycuda import characterize
from pycuda.characterize import sizeof
from pycuda.compiler import SourceModule
import numpy as np
import time

MyGraph = GraphColor("net50k001.txt")

# TODO: implementer methode MCMC

g = pycuda.curandom.XORWOWRandomNumberGenerator()
nb_nodes = MyGraph.graphStruct.number_of_nodes()
seed = int(time.time())

threadPerBlock = (32, 1, 1)
BlockPerGrid = ((nb_nodes + threadPerBlock[0] - 1)/threadPerBlock[0], 1, 1)
rand_states = cuda.mem_alloc(nb_nodes*characterize.sizeof('curandStateXORWOW', '#include <curand_kernel.h>'))

with open('CUDA/initializeCuda.cu', 'r') as myfile:
    cuda_code = myfile.read()

mod= SourceModule(cuda_code, no_extern_c=True)
func_initCurand = mod.get_function("initCurand")
func_initCurand(rand_states, np.uint32(seed), np.uint32(nb_nodes), block=threadPerBlock, grid=BlockPerGrid)
