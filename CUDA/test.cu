#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

extern "C" {
    __global__ void testRand( curandState * state, int nb ){
        int id = threadIdx.x  + blockIdx.x * blockDim.x;
        int value;
        for (int i=0;i<nb;i++){
            curandState localState = state[id];
            value = curand(&localState);
            //state[id] = localState;
            printf("Id %i, value %i\n",id,value);
        }
    }
    __global__ void setup_kernel( curandState * state, unsigned long seed )
    {
        int id = threadIdx.x  + blockIdx.x * blockDim.x;
        curand_init( seed, id , 0, &state[id] );
    }
}