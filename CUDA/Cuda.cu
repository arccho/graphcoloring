#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

typedef uint32_t col_sz;     // node color
typedef uint32_t node;     // graph node
typedef uint32_t node_sz;

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

    /**
    * Parallel sum reduction inside a single warp
    */
    __device__ void warpReduction(volatile float *sdata, uint32_t tid, uint32_t blockSize) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
        if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
        if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
        if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
    }

    __global__ void sumReduction(uint32_t nedges, float * conflictCounter_d) {

        uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

        if (idx >= nedges)
            return;

        extern	__shared__ float sdata[];

        uint32_t tid = threadIdx.x;
        uint32_t blockSize = blockDim.x;
        uint32_t i = (blockSize * 2) * blockIdx.x + tid;

        sdata[tid] = conflictCounter_d[i] + conflictCounter_d[i + blockSize];

        __syncthreads();

        //useless for blocks of dim <= 64
        if (blockSize >= 512)
        {
            if (tid < 256)
                sdata[tid] += sdata[tid + 256];
            __syncthreads();
        }
        if (blockSize >= 256)
        {
            if (tid < 128)
                sdata[tid] += sdata[tid + 128];
            __syncthreads();
        }
        if (blockSize >= 128)
        {
            if (tid < 64)
                sdata[tid] += sdata[tid + 64];
            __syncthreads();
        }

        if (tid < 32)
            //warpReduction<blockSize>(sdata, tid);
            warpReduction(sdata, tid, blockSize);

        if (tid == 0)
            conflictCounter_d[blockIdx.x] = sdata[0];
    }

    __global__ void selectStarColoring(uint32_t nnodes, uint32_t * starColoring_d, float * qStar_d, col_sz nCol, uint32_t * coloring_d, node_sz * cumulDegs, node * neighs, bool * colorsChecker_d, uint32_t * orderedColors_d, curandState * states, float epsilon, uint32_t * statsFreeColors_d) {

        uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;

        if (idx >= nnodes)
            return;

        uint32_t index = cumulDegs[idx];							//index of the node in neighs
        uint32_t nneighs = cumulDegs[idx + 1] - index;				//number of neighbors

        uint32_t nodeCol = coloring_d[idx];							//node color

        bool * colorsChecker = &(colorsChecker_d[idx * nCol]);		//array used to set to 1 or 0 the colors occupied from the neighbors
        for (int i = 0; i < nneighs; i++)
            colorsChecker[coloring_d[neighs[index + i]]] = 1;

        uint32_t * orderedColors = &(orderedColors_d[idx * nCol]);	//array containing previously occupied colors and then free ones
        uint32_t Zp = nCol, Zn = 0;									//number of free colors (p) and occupied colors (n)
        for (int i = 0; i < nCol; i++)
        {
            orderedColors[Zn] += i * (1 - (1 - colorsChecker[i]));
            orderedColors[Zp - 1] += i * (1 - colorsChecker[i]);
            Zn += colorsChecker[i];
            Zp -= 1 - colorsChecker[i];
        }
        Zp = nCol - Zn;

    #ifdef STATS
        statsFreeColors_d[idx] = Zp;
    #endif // STATS

        if (!Zp)													//manage exception of no free colors
        {
    #ifdef FIXED_N_COLORS
            starColoring_d[idx] = nodeCol;
            qStar_d[idx] = 1;
    #endif // FIXED_N_COLORS
    #ifdef DYNAMIC_N_COLORS
            starColoring_d[idx] = nodeCol;
            qStar_d[idx] = 1;
    #endif // DYNAMIC_N_COLORS
            return;
        }

        float randnum = curand_uniform(&states[idx]);				//random number

        float threshold;
        uint32_t selectedIndex = 0;									//selected index for orderedColors to select the new color
        if (colorsChecker[nodeCol])									//if node color is used by neighbors
        {
            threshold = 1 - epsilon * Zn;							//threshold used to randomly determine whether to extract a free color or a busy one
            if (randnum < threshold)
            {
                selectedIndex = ((randnum * Zp) / threshold) + Zn;	//get the selected index
                qStar_d[idx] = (1 - epsilon * Zn) / Zp;				//save the probability of the color chosen
            }
            else
            {
                selectedIndex = ((randnum - threshold) * Zn) / (1 - threshold);	//get the selected index
                qStar_d[idx] = epsilon;								//save the probability of the color chosen
            }
            starColoring_d[idx] = orderedColors[selectedIndex];		//save the new color
        }
        else
        {
            threshold = 1 - epsilon * (nCol - 1);					//threshold used to randomly determine whether to extract a occupied color
                                                                    //or keep the same
            if (randnum < threshold)
            {
                starColoring_d[idx] = nodeCol;						//keep the same color
                qStar_d[idx] = 1 - ((nCol - 1) * epsilon);	//save the probability of the color chosen
            }
            else
            {
                selectedIndex = ((randnum - threshold) * Zn) / (1 - threshold);	//get the selected index
                starColoring_d[idx] = orderedColors[selectedIndex];	//save the new color
                qStar_d[idx] = epsilon;						//save the probability of the color chosen
            }
        }
    }
}

