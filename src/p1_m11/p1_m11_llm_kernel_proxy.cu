#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK_CUDA(call)                                                         \
do {                                                                             \
    cudaError_t err = (call);                                                    \
    if (err != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA ERROR %s:%d: %s\n",                                \
                __FILE__, __LINE__, cudaGetErrorString(err));                    \
        exit(1);                                                                 \
    }                                                                            \
} while (0)


__global__ void kv_copy_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        dst[i] = src[i];
    }
}


__global__ void rmsnorm_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int hidden,
    float eps
) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* row_x = x + row * hidden;
    float* row_y = y + row * hidden;

    float local = 0.0f;

    for (int i = tid; i < hidden; i += blockDim.x) {
        float v = row_x[i];
        local += v * v;
    }

    smem[tid] = local;
    __syncthreads();

    for (
        int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }

        __syncthreads();
    }

    float inv_rms = rsqrtf(
        smem[0] / hidden + eps
    );

    for (int i = tid; i < hidden; i += blockDim.x) {
        row_y[i] = row_x[i] * inv_rms;
    }
}


__global__ void softmax_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int width
) {
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    const float* row_x = x + row * width;
    float* row_y = y + row * width;

    float local_max = -INFINITY;

    for (int i = tid; i < width; i += blockDim.x) {
        local_max = fmaxf(
            local_max,
            row_x[i]
        );
    }

    smem[tid] = local_max;
    __syncthreads();

    for (
        int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (tid < stride) {
            smem[tid] = fmaxf(
                smem[tid],
                smem[tid + stride]
            );
        }

        __syncthreads();
    }

    float row_max = smem[0];

    float local_sum = 0.0f;

    for (int i = tid; i < width; i += blockDim.x) {
        local_sum += expf(
            row_x[i] - row_max
        );
    }

    smem[tid] = local_sum;
    __syncthreads();

    for (
        int stride = blockDim.x / 2;
        stride > 0;
        stride >>= 1
    ) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }

        __syncthreads();
    }

    float inv_sum = 1.0f / smem[0];

    for (int i = tid; i < width; i += blockDim.x) {
        row_y[i] = expf(
            row_x[i] - row_max
        ) * inv_sum;
    }
}


/*
    Simple FP32 tiled GEMM.

    This is NOT intended to beat cuBLAS.

    Its purpose is to provide a transparent,
    deterministic compute-heavy kernel whose
    profiling characteristics can be contrasted
    against memory-oriented kernels.
*/

template<int TILE>
__global__ void tiled_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;

    for (
        int tile = 0;
        tile < N;
        tile += TILE
    ) {
        int a_col = tile + threadIdx.x;
        int b_row = tile + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (
                row < N
                && a_col < N
            )
            ? A[row * N + a_col]
            : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (
                b_row < N
                && col < N
            )
            ? B[b_row * N + col]
            : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += (
                As[threadIdx.y][k]
                *
                Bs[k][threadIdx.x]
            );
        }

        __syncthreads();
    }

    if (
        row < N
        && col < N
    ) {
        C[row * N + col] = sum;
    }
}


static float time_kernel_start() {
    return 0.0f;
}


int main() {
    cudaDeviceProp prop{};

    CHECK_CUDA(
        cudaGetDeviceProperties(
            &prop,
            0
        )
    );

    printf("GPU = %s\n", prop.name);
    printf(
        "COMPUTE_CAPABILITY = %d.%d\n",
        prop.major,
        prop.minor
    );

    /*
        Workload sizes are deliberately large enough
        to produce stable measurable kernels while
        staying well inside RTX 4090 memory.
    */

    const int KV_N = 64 * 1024 * 1024;

    const int ROWS = 4096;
    const int HIDDEN = 4096;

    const int SOFTMAX_ROWS = 8192;
    const int SOFTMAX_WIDTH = 2048;

    const int GEMM_N = 2048;


    float *kv_src, *kv_dst;

    CHECK_CUDA(
        cudaMalloc(
            &kv_src,
            KV_N * sizeof(float)
        )
    );

    CHECK_CUDA(
        cudaMalloc(
            &kv_dst,
            KV_N * sizeof(float)
        )
    );


    float *norm_x, *norm_y;

    CHECK_CUDA(
        cudaMalloc(
            &norm_x,
            (size_t)ROWS
            * HIDDEN
            * sizeof(float)
        )
    );

    CHECK_CUDA(
        cudaMalloc(
            &norm_y,
            (size_t)ROWS
            * HIDDEN
            * sizeof(float)
        )
    );


    float *soft_x, *soft_y;

    CHECK_CUDA(
        cudaMalloc(
            &soft_x,
            (size_t)SOFTMAX_ROWS
            * SOFTMAX_WIDTH
            * sizeof(float)
        )
    );

    CHECK_CUDA(
        cudaMalloc(
            &soft_y,
            (size_t)SOFTMAX_ROWS
            * SOFTMAX_WIDTH
            * sizeof(float)
        )
    );


    float *A, *B, *C;

    CHECK_CUDA(
        cudaMalloc(
            &A,
            (size_t)GEMM_N
            * GEMM_N
            * sizeof(float)
        )
    );

    CHECK_CUDA(
        cudaMalloc(
            &B,
            (size_t)GEMM_N
            * GEMM_N
            * sizeof(float)
        )
    );

    CHECK_CUDA(
        cudaMalloc(
            &C,
            (size_t)GEMM_N
            * GEMM_N
            * sizeof(float)
        )
    );


    CHECK_CUDA(
        cudaMemset(
            kv_src,
            1,
            KV_N * sizeof(float)
        )
    );

    CHECK_CUDA(
        cudaMemset(
            norm_x,
            1,
            (size_t)ROWS
            * HIDDEN
            * sizeof(float)
        )
    );

    CHECK_CUDA(
        cudaMemset(
            soft_x,
            1,
            (size_t)SOFTMAX_ROWS
            * SOFTMAX_WIDTH
            * sizeof(float)
        )
    );

    CHECK_CUDA(
        cudaMemset(
            A,
            1,
            (size_t)GEMM_N
            * GEMM_N
            * sizeof(float)
        )
    );

    CHECK_CUDA(
        cudaMemset(
            B,
            1,
            (size_t)GEMM_N
            * GEMM_N
            * sizeof(float)
        )
    );


    dim3 copy_block(256);

    dim3 copy_grid(
        (KV_N + 255) / 256
    );


    dim3 norm_block(256);

    dim3 norm_grid(
        ROWS
    );


    dim3 soft_block(256);

    dim3 soft_grid(
        SOFTMAX_ROWS
    );


    constexpr int TILE = 16;

    dim3 gemm_block(
        TILE,
        TILE
    );

    dim3 gemm_grid(
        (GEMM_N + TILE - 1) / TILE,
        (GEMM_N + TILE - 1) / TILE
    );


    /*
        Warm-up.
    */

    kv_copy_kernel<<<
        copy_grid,
        copy_block
    >>>(
        kv_src,
        kv_dst,
        KV_N
    );

    rmsnorm_kernel<<<
        norm_grid,
        norm_block,
        256 * sizeof(float)
    >>>(
        norm_x,
        norm_y,
        HIDDEN,
        1e-6f
    );

    softmax_kernel<<<
        soft_grid,
        soft_block,
        256 * sizeof(float)
    >>>(
        soft_x,
        soft_y,
        SOFTMAX_WIDTH
    );

    tiled_gemm_kernel<TILE><<<
        gemm_grid,
        gemm_block
    >>>(
        A,
        B,
        C,
        GEMM_N
    );

    CHECK_CUDA(
        cudaDeviceSynchronize()
    );


    printf(
        "WARMUP = SUCCESS\n"
    );


    /*
        Measured launches.

        Repeated a few times so NCU has deterministic
        kernels available for collection.
    */

    for (int repeat = 0; repeat < 3; ++repeat) {

        kv_copy_kernel<<<
            copy_grid,
            copy_block
        >>>(
            kv_src,
            kv_dst,
            KV_N
        );

        rmsnorm_kernel<<<
            norm_grid,
            norm_block,
            256 * sizeof(float)
        >>>(
            norm_x,
            norm_y,
            HIDDEN,
            1e-6f
        );

        softmax_kernel<<<
            soft_grid,
            soft_block,
            256 * sizeof(float)
        >>>(
            soft_x,
            soft_y,
            SOFTMAX_WIDTH
        );

        tiled_gemm_kernel<TILE><<<
            gemm_grid,
            gemm_block
        >>>(
            A,
            B,
            C,
            GEMM_N
        );
    }


    CHECK_CUDA(
        cudaDeviceSynchronize()
    );


    printf(
        "MEASURED_LAUNCHES = 12\n"
    );

    printf(
        "KERNEL_KV_COPY = SUCCESS\n"
    );

    printf(
        "KERNEL_RMSNORM = SUCCESS\n"
    );

    printf(
        "KERNEL_SOFTMAX = SUCCESS\n"
    );

    printf(
        "KERNEL_TILED_GEMM = SUCCESS\n"
    );


    cudaFree(kv_src);
    cudaFree(kv_dst);

    cudaFree(norm_x);
    cudaFree(norm_y);

    cudaFree(soft_x);
    cudaFree(soft_y);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);


    return 0;
}
