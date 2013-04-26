#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <algorithm>

#include "GameOfLife.h"

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error: %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

__device__ short getLivingNeighbouringCells(bool state[], const int phase, const int index, const int width, const int height) {
    short livingCells = 0;
    const int offset = phase * width * height;
    int icLeft = 0, icRight = 0, icBottom = 0, icTop = 0;

    // Creating an infinite field
    if ((index % width) == 0) { // On the left side
        icLeft = width;
    } else if ((index % (width - 1)) == 0) { // On the right side
        icRight = -(width - 1);
    }
    if (index >= 0 && index <= width) { // On the top
        icTop = (height - 1) * width;
    } else if (index >= (width * height) - width && index <= width * height) { // On the bottom
        icBottom = -((height - 1) * width);
    }

    livingCells += state[offset + index - 1 + icLeft] + state[offset + index + 1 + icRight];
    livingCells += state[offset + index - width + icTop] + state[offset + index - width - 1 + icLeft + icTop] + state[offset + index - width + 1 + icRight + icTop];
    livingCells += state[offset + index + width + icBottom] + state[offset + index + width - 1 + icLeft + icBottom] + state[offset + index + width + 1 + icRight + icBottom];

    return livingCells;
}

__global__ void simulateGameOfLife(bool states[], const int steps, const int start, const int width, const int height) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int N = width * height;

    if (i >= N) {
        return;
    }

    for (int j = start; j < steps - 1; ++j) { // less and not lte, because we compute j+1
        const short cells = getLivingNeighbouringCells(states, j, i, width, height);
        if (cells == 2) {
            states[N * (j + 1) + i] = states[(N * j) + i];
        } else if (cells < 2 || cells > 3) {
            states[N * (j + 1) + i] = false;
        } else {
            states[N * (j + 1) + i] = true;
        }
    }
}

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {
    const int width = 600;
    const int height = 600;
    const int steps = 1500;
    const int totalStreams = 5;

    // Computed values
    const int N = width * height;
    const int threadsPerBlock = 128;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    const int stepsForOneKernel = steps / totalStreams;
    const size_t copySizeForOneKernel = stepsForOneKernel * N * sizeof(bool);

    std::clog << "Threads per block: " << threadsPerBlock << std::endl;
    std::clog << "Blocks per grid: " << blocksPerGrid << std::endl;
    std::clog << "Steps for a kernel: " << stepsForOneKernel << std::endl;

    cudaStream_t streams[totalStreams];
    cudaEvent_t events[totalStreams];
    for (int i = 0; i < totalStreams; ++i) {
        CUDA_CHECK_RETURN(cudaStreamCreate(&streams[i]));
        CUDA_CHECK_RETURN(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
    }

    // Device allocation
    bool *d_states;
    size_t totalSize = N * sizeof(bool) * steps;
    CUDA_CHECK_RETURN(cudaMalloc(&d_states, totalSize));

    // Host allocation
    bool *h_states;
    CUDA_CHECK_RETURN(cudaMallocHost(&h_states, totalSize));
    createRandomCells(h_states, width, height);

    // Host to device copy the first random step
    size_t sizeFirstStep = width * height * sizeof(bool);
    CUDA_CHECK_RETURN(cudaMemcpy(d_states, h_states, sizeFirstStep, cudaMemcpyHostToDevice));

    // Measure timing
    cudaEvent_t cu_start, cu_end;
    float elapsedTime;
    CUDA_CHECK_RETURN(cudaEventCreateWithFlags(&cu_start, cudaEventBlockingSync));
    CUDA_CHECK_RETURN(cudaEventCreateWithFlags(&cu_end, cudaEventBlockingSync));
    CUDA_CHECK_RETURN(cudaEventRecord(cu_start, 0));

    for (int i = 0; i < totalStreams; ++i) {
        const int start = i * stepsForOneKernel;
        const size_t offset = start * sizeof(bool);

        if (i > 0) {
            CUDA_CHECK_RETURN(cudaEventSynchronize(events[i - 1]));
        }

        simulateGameOfLife<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(d_states, (i + 1) * stepsForOneKernel, start , width, height);
        CUDA_CHECK_RETURN(cudaEventRecord(events[i], streams[i]));
        CUDA_CHECK_RETURN(cudaMemcpyAsync(h_states + offset, d_states + offset, copySizeForOneKernel, cudaMemcpyDeviceToHost, streams[i]));
    }

    CUDA_CHECK_RETURN(cudaEventRecord(cu_end, 0));
    CUDA_CHECK_RETURN(cudaEventSynchronize(cu_end));
    CUDA_CHECK_RETURN(cudaEventElapsedTime(&elapsedTime, cu_start, cu_end));

    std::cout << (elapsedTime / 1000) << " seconds were needed to compute this Game of Life table" << std::endl;

    writeResultToFile("/home/aki/temp/gol.lif", h_states, width, height, steps);

    // Freeing resources
    CUDA_CHECK_RETURN(cudaFree(d_states));
    CUDA_CHECK_RETURN(cudaFreeHost(h_states));
    CUDA_CHECK_RETURN(cudaEventDestroy(cu_start));
    CUDA_CHECK_RETURN(cudaEventDestroy(cu_end));
    CUDA_CHECK_RETURN(cudaDeviceReset());

    return 0;
}
