#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <cuda_runtime.h>

#define M 1024 // 表示左矩阵（A矩阵）的行数。
#define N 1024 // 表示右矩阵（B矩阵）的列数。
#define K 1024 // 表示左矩阵（A矩阵）的列数，同时也是右矩阵（B矩阵）的行数。
#define THREADS 16
#define BM 128 // 每个Block负责计算矩阵C中的BM * BN个元素
#define BN 128
#define K 8
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

__global__ void mxm_shared(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (ty < M && tx < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += a[ty * K + i] * b[i * N + tx];
        }
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
    mxm_naive<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);
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
