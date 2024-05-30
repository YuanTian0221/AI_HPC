#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// 定义矩阵大小
#define M 1024
#define N 1024
#define K 1024

int main() {
    // 定义 cuBLAS 句柄
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 分配主机内存
    float *h_A = (float *)malloc(M * K * sizeof(float));
    float *h_B = (float *)malloc(K * N * sizeof(float));
    float *h_C = (float *)malloc(M * N * sizeof(float));

    // 初始化矩阵 A 和 B
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = 1.0f;
    }

    for (int i = 0; i < K * N; ++i) {
        h_B[i] = 1.0f;
    }

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, M * K * sizeof(float));
    cudaMalloc((void **)&d_B, K * N * sizeof(float));
    cudaMalloc((void **)&d_C, M * N * sizeof(float));

    // 将数据从主机内存复制到设备内存
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始事件
    cudaEventRecord(start, 0);

    // 执行矩阵相乘操作
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);

    // 记录结束事件
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // 计算执行时间
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Elapsed time: %f ms\n", elapsedTime);

    // 将结果从设备内存复制回主机内存
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果的一部分
    for (int i = 0; i < 10; ++i) {
        printf("%f ", h_C[i]);
    }
    printf("\n");

    // 释放内存
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 销毁 cuBLAS 句柄
    cublasDestroy(handle);

    return 0;
}
