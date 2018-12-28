import pycuda.driver as cuda
from pycuda import autoinit
from pycuda.characterize import sizeof
import numpy
import time
from pycuda.compiler import SourceModule


MAXTHREADS = 2
NBBLOCKS = 2

#ouvre un fichier et le charge dans un string
with open('CUDA/test.cu', 'r') as myfile:
        cuda_code = myfile.read()
mod= SourceModule(cuda_code, no_extern_c=True)


blockSize = (MAXTHREADS, 1, 1)
gridSize = (NBBLOCKS, 1, 1)

devStates = cuda.mem_alloc(MAXTHREADS*NBBLOCKS*sizeof('curandState', '#include <curand_kernel.h>'))
t = int(time.time())
func_setup_kernel = mod.get_function("setup_kernel")
func_setup_kernel(devStates, numpy.int64(t), block=blockSize, grid=gridSize)

nb = 4
func_testRand = mod.get_function("testRand")
func_testRand(devStates, numpy.int32(nb), block=blockSize, grid=gridSize)
func_testRand(devStates, numpy.int32(nb), block=blockSize, grid=gridSize)