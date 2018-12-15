#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


extern "C" {

 __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    //printf("%d ", idx);
    //printf("%d ", a[idx]);
    a[idx] *= 2;
  }

  __global__ void initCurand(curandState* states, uint32_t seed, uint32_t nElem ) {
	//uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
	//if (tid < nElem) {
	//	curand_init( seed, tid, 0, &states[tid] );
	//}
}



}

