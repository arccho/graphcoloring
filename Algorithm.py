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
nb_edges = MyGraph.graphStruct.number_of_edges()
seed = int(time.time())

threadPerBlock = (32, 1, 1)
BlockPerGrid = ((nb_nodes + threadPerBlock[0] - 1)/threadPerBlock[0], 1, 1)
rand_states = cuda.mem_alloc(nb_nodes*characterize.sizeof('curandStateXORWOW', '#include <curand_kernel.h>'))

with open('CUDA/initializeCuda.cu', 'r') as myfile:
    cuda_code = myfile.read()

mod= SourceModule(cuda_code, no_extern_c=True)
func_initCurand = mod.get_function("initCurand")
func_initCurand(rand_states, np.uint32(seed), np.uint32(nb_nodes), block=threadPerBlock, grid=BlockPerGrid, time_kernel = True)

#params
p_nb_col = MyGraph.maxDeg
p_startingNCol = 50
p_epsilon = 1e-8
p_lambda = 0.01
p_ratioFreezed = 1e-2
p_maxRip = 250
p_numThreads = 32

#configuration grille
threadsPerBlock = (p_numThreads, 1, 1)
blocksPerGrid = ((nb_nodes + p_numThreads -1)/ p_numThreads, 1,1)
blocksPerGrid_nCol = ((p_nb_col + threadsPerBlock[0] -1)/threadsPerBlock[0], 1, 1)
blocksPerGrid_half = (((nb_nodes / 2) + threadsPerBlock[0] - 1) / threadsPerBlock[0], 1, 1)
blocksPerGrid_edges = ((nb_edges + threadsPerBlock[0] - 1) / threadsPerBlock[0], 1, 1)
blocksPerGrid_half_edges = (((nb_edges / 2) + threadsPerBlock[0] - 1) / threadsPerBlock[0], 1, 1)


##############################################
sizeof_uint32 = sizeof("uint32_t", "#include <stdint.h>")
sizeof_float = sizeof("float")
sizeof_bool = sizeof("bool")

free_mem, tot_mem = cuda.mem_get_info()
print "total mem: " + str(tot_mem) + " free mem: " + str(free_mem)

tot = nb_nodes * sizeof_uint32 * 3
print "nb_nodes * sizeof(uint32_t): " + str(nb_nodes * sizeof_uint32) + " x3"
tot = tot + nb_nodes * sizeof_float * 2
print "nb_nodes * sizeof(np.float32): " + str(nb_nodes * sizeof_float) + " x2"
tot = tot + nb_edges * sizeof_uint32
print "nb_edges * sizeof(np.uint32): " + str(nb_edges * sizeof_uint32) + " x1"
tot = tot + nb_nodes * p_nb_col * sizeof_bool
print "nb_nodes * p_nb_col * sizeof(np.bool): " + str(nb_nodes * p_nb_col * sizeof_bool) + " x1"
tot = tot + nb_nodes * p_nb_col * sizeof_uint32
print "nb_nodes * p_nb_col * sizeof(np.uint32): " + str(nb_nodes * p_nb_col * sizeof_uint32) + " x1"
print "TOTAL: " + str(tot) + " bytes"

###############################################################
#Cuda Allocation
coloring_d = cuda.mem_alloc(nb_nodes * sizeof_uint32)
starColoring_d = cuda.mem_alloc(nb_nodes * sizeof_uint32)
q_h = np.zeros(nb_nodes * sizeof_float, dtype=np.float32)
q_d = cuda.mem_alloc(nb_nodes * sizeof_float)
qStar_h = np.zeros(nb_nodes * sizeof_float, dtype=np.float32)
qStar_d = cuda.mem_alloc(nb_nodes * sizeof_float)
conflictCounter_h = np.zeros(nb_edges * sizeof_uint32, dtype=np.uint32)
conflictCounter_d = cuda.mem_alloc(nb_edges * sizeof_uint32)
colorsChecker_d = cuda.mem_alloc(nb_edges * p_nb_col * sizeof_bool)
#print "colorsChecker_d: " + str(nb_nodes * p_nb_col * sizeof_bool)

#ifdef STANDARD
orderedColors_d = cuda.mem_alloc(nb_nodes * p_nb_col * sizeof_uint32)
#endif // STANDARD

free_mem, tot_mem = cuda.mem_get_info()
print "total mem: " + str(tot_mem) + " free mem: " + str(free_mem)

#ifdef PRINTS
print "ColoringMCMC GPU"
print "nbCol: " +str(p_nb_col)
print "epsilon: " + str(p_epsilon)
print "lambda: " + str(p_lambda)
print "ratioFreezed: " + str(p_ratioFreezed)
print "maxRip: " + str(p_maxRip)
#endif // PRINTS

#ifdef WRITE
logFile = open(str(nb_nodes) + "-" + str(nb_edges) + "-logFile.txt" , "wt")
resultsFile = open(str(nb_nodes) + "-" + str(nb_edges) + "-resultsFile.txt" , "wt")
colorsFile = open(str(nb_nodes) + "-" + str(nb_edges) + "-colorsFile.txt" , "wt")

logFile.write("nbCol: " + str(p_nb_col) + "\n")
logFile.write("epsilon: " + str(p_epsilon) + "\n")
logFile.write("lambda: " + str(p_lambda) + "\n")
logFile.write("ratioFreezed: " + str(p_ratioFreezed) + "\n")
logFile.write("maxRip: " + str(p_maxRip) + "\n")

resultsFile.write("nbCol: " + str(p_nb_col) + "\n")
resultsFile.write("epsilon: " + str(p_epsilon) + "\n")
resultsFile.write("lambda: " + str(p_lambda) + "\n")
resultsFile.write("ratioFreezed: " + str(p_ratioFreezed) + "\n")
resultsFile.write("maxRip: " + str(p_maxRip) + "\n")
#endif // WRITE







