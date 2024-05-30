#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <assert.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define M 512 //表示左矩阵（A矩阵）的行数。
#define N 512 //表示右矩阵（B矩阵）的列数。
#define K 512 //表示左矩阵（A矩阵）的列数，同时也是右矩阵（B矩阵）的行数。
#define BLOCKS 32
#define THREADS 512
#define NUM_STREAMS 4

__global__ void mxm_naive(float *a, float *b, float *c, int M, int N, int K){
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (ty<M && TX<N){
        float sum = 0.0;
        for(int i=0;i<K;i++){
            sum+=a[ty*K+i]*b[i*N+tx];
        }
        c[ty*N+tx] = sum;
    }
}

int main(){
    // 定义变量
    float *a,*b,*c;
    float *d_a,*d_b,*d_c;
    // 定义函数数组
    void (*kernel[NUM_STREAMS])();
    // 函数数组初始化
    kernel[0] = mxm_naive;

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
        a[i] = i;
    }
    for (int i = 0; i < K*N; ++i) {
        b[i] = i;
    }
    // 数据复制到gpu
    assert(cudaMemcpy(d_a,a,M*K*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess);
    assert(cudaMemcpy(d_b,b,K*N*sizeof(double),cudaMemcpyHostToDevice)==cudaSuccess);
    
    // Define CUDA Event
    cudaEvent_t start_device[NUM_STREAMS],stop_device[NUM_STREAMS];   /* CUDA timers */
    cudaStream_t streams[NUM_STREAMS];
    float time_device[NUM_STREAMS];
    // Create CUDA Event and Streams
    for(int s=0;s<NUM_STREAMS;s++){
        cudaEventCreate(&start_device[s]);
        cudaEventCreate(&stop_device[s]);
        cudaStreamCreate(&streams[s]);
    }
    
    // Define grid and block dimensions
    dim3 dimBlock(THREADS, THREADS);
    dim3 dimGrid((M + THREADS - 1) / THREADS, (N + THREADS - 1) / THREADS); 
    // Lanuch Kernel
    for(int s=0;s<NUM_STREAMS;s++){
        // Record Time
        cudaEventRecord(start_device[s], streams[s]);
        // Kernel Begin
        kernel[s]<<<dimGrid,dimBlock,streams[s]>>>(d_a,d_b,d_c,N);
    }
    
    // Kernel End
    for(int s=0;s<NUM_STREAMS;s++){
        cudaEventSynchronize(stop_device[s]);
        cudaEventRecord(stop_device[s], streams[s]);
        cudaStreamSynchronize(streams[s]);
        cudaEventElapsedTime(&time_device[s], start_devic[s], stop_devic[s]);
    }
    // CUDA Event Streams Destroy
    for(int s=0;s<NUM_STREAMS;s++){
        cudaEventDestroy(start_devic[s]);
        cudaEventDestroy(stop_devic[s]);
        cudaStreamDestroy(streams[s]);
    }
    
    // cudaMemcpyDeviceToHost
    cudaMemcpy(c,d_c,M*N*sizeof(float),cudaMemcpyDeviceToHost);
    // Free cuda memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceReset();
    return 0;
}