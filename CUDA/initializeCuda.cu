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

    __global__ void conflictChecker(uint32_t nedges, uint32_t * conflictCounter_d, uint32_t * coloring_d, uint32_t * edges) {

        uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

        if (idx >= nedges)
            return;

        uint32_t idx0 = idx * 2;
        uint32_t idx1 = idx0 + 1;

        uint32_t node0 = edges[idx0];
        uint32_t node1 = edges[idx1];

        uint32_t col0 = coloring_d[node0];
        uint32_t col1 = coloring_d[node1];

        conflictCounter_d[idx] = col0 == col1;
    }
}

