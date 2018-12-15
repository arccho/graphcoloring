import pycuda.autoinit
import pycuda.driver as cuda
import numpy
from pycuda.compiler import SourceModule

#ouvre un fichier et le charge dans un string
with open('CUDA/initializeCuda.cu', 'r') as myfile:
        cuda_code = myfile.read()

#
mod= SourceModule(cuda_code)
func_doublify = mod.get_function("doublify")

a = numpy.random.randn(4,4)
#print a
a = a.astype(numpy.float32)
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu, a)

func_doublify = mod.get_function("doublify")
func_doublify(a_gpu, block=(4,4,1))
cuda.memcpy_dtoh(a, a_gpu)

#print a

#######################################################


mod = SourceModule("""
    #include <stdio.h>

    __global__ void say_hi()
    {
      //printf("I am %dth thread in threadIdx.x:%d.threadIdx.y:%d  blockIdx.:%d blockIdx.y:%d blockDim.x:%d blockDim.y:%d\\n",(threadIdx.x+threadIdx.y*blockDim.x+(blockIdx.x*blockDim.x*blockDim.y)+(blockIdx.y*blockDim.x*blockDim.y)),threadIdx.x, threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);
    }
    """)

func = mod.get_function("say_hi")
func(block=(4,4,1),grid=(2,2,1))

################################################
mod = SourceModule("""
    #include <stdio.h>

    __global__ void increment(int* resultat)
    {
      atomicAdd(resultat, 1);
      atomicSub(resultat, 1);
      atomicAdd(resultat, 1);
      atomicSub(resultat, 1);
            atomicAdd(resultat, 1);
      atomicSub(resultat, 1);
            atomicAdd(resultat, 1);

    }
    """)

func = mod.get_function("increment")
tot = numpy.zeros(1)
print tot
tot= tot.astype(numpy.int64)
tot_gpu = cuda.mem_alloc(tot.nbytes)
cuda.memcpy_htod(tot_gpu, tot)
func(tot_gpu, block=(8,8,16),grid=(128,128,128)) #Dimension de chaque bloc: 4x4 thread en 2D & Dimension de la grille: 2x2blocs en 2D
cuda.memcpy_dtoh(tot, tot_gpu)
print tot