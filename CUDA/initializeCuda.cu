#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


extern "C" {

    __global__ void initCurand(curandState* states, uint32_t seed, uint32_t nElem ) {
	    uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
	    if (tid < nElem) {
	        states[tid] = curandState();
	        curand_init( seed, tid, 0, &states[tid] );
	    }
    }

    __global__ void initColoring(uint32_t nnodes, uint32_t * coloring_d, float nCol, curandState * states, uint32_t seed) {

	    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	    if (idx >= nnodes)
		    return;

	    float randnum = curand_uniform(&states[idx]);

	    int color = (int)(randnum * nCol);
	    //printf("color=%d\n", states[idx].d);

	    coloring_d[idx] = color;
	    //coloring_d[idx] = 0;
    }


}

