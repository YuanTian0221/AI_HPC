#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>

#define M 1024 // 表示左矩阵（A矩阵）的行数。
#define N 1024 // 表示右矩阵（B矩阵）的列数。
#define K 1024 // 表示左矩阵（A矩阵）的列数，同时也是右矩阵（B矩阵）的行数。
#define THREADS 16

#define TILE_WIDTH 32 // 定义共享内存块大小

__global__ void mxm_shared(float *a, float *b, float *c) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    // 定义共享内存数组
    __shared__ float shared_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_b[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    // 遍历矩阵 a 和 b 的所有分块
    for (int i = 0; i < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
        // 将矩阵 a 的分块加载到共享内存
        if (ty < M && i * BLOCK_SIZE + threadIdx.x < K) {
            shared_a[threadIdx.y][threadIdx.x] = a[ty * K + i * BLOCK_SIZE + threadIdx.x];
        } else {
            shared_a[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // 将矩阵 b 的分块加载到共享内存
        if (tx < N && i * BLOCK_SIZE + threadIdx.y < K) {
            shared_b[threadIdx.y][threadIdx.x] = b[(i * BLOCK_SIZE + threadIdx.y) * N + tx];
        } else {
            shared_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 等待所有线程加载完毕
        __syncthreads();

        // 计算局部矩阵乘积的部分
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            sum += shared_a[threadIdx.y][j] * shared_b[j][threadIdx.x];
        }

        // 等待所有线程完成计算
        __syncthreads();
    }

    // 将计算结果写入全局内存
    if (ty < M && tx < N) {
        c[ty * N + tx] = sum;
    }
}


int main(){
    // 定义变量
    float *a,*b,*c;
    float *d_a,*d_b,*d_c;

    // cpu分配空间
    a = new float[M*K];
    b = new float[K*N];
    c = new float[M*N];
    // gpu分配空间
    assert(cudaMalloc((void**)&d_a, M*K * sizeof(float))==cudaSuccess);
    assert(cudaMalloc((void**)&d_b, K*N * sizeof(float))==cudaSuccess);
    assert(cudaMalloc((void**)&d_c, M*N * sizeof(float))==cudaSuccess);
    // 初始化数据
    for (int i = 0; i < M*K; ++i) {
        a[i] = 1.0f;
    }
    for (int i = 0; i < K*N; ++i) {
        b[i] = 1.0f;
    }
    // 数据复制到gpu
    assert(cudaMemcpy(d_a, a, M*K*sizeof(float), cudaMemcpyHostToDevice)==cudaSuccess);
    assert(cudaMemcpy(d_b, b, K*N*sizeof(float), cudaMemcpyHostToDevice)==cudaSuccess);
    
    // Define grid and block dimensions
    dim3 dimBlock(THREADS, THREADS);
    dim3 dimGrid((N + THREADS - 1) / THREADS, (M + THREADS - 1) / THREADS); 
    
    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);
    // Launch Kernel
    mxm_shared<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);
    // Record the stop event
    cudaEventRecord(stop, 0);

    // Synchronize the stop event
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    // cudaMemcpyDeviceToHost
    cudaMemcpy(c, d_c, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // Print the elapsed time
    printf("dimGrid = nBlocks: %d * nBlocks: %d, dimBlock = nThreads %d * nThreads %d.\n", (N + THREADS - 1) / THREADS, (M + THREADS - 1) / THREADS, THREADS, THREADS);
    printf("Elapsed time: %f ms\n", elapsedTime);

    // Print part of the result
    for (int i = 0; i < 10; ++i) {
        printf("%f ", c[i]);
    }
    printf("\n");
    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free cuda memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceReset();
    return 0;
}
