#!/usr/bin/env bash
set -uo pipefail

BASE="/root/autodl-tmp/ai-infra"
PROJECT="${BASE}/project1-turbollm"

ENV="${BASE}/envs/project1-vllm-0.27.1"
PYTHON="${ENV}/bin/python"

SRC_DIR="${PROJECT}/src/p1_m11"
RESULT_DIR="${PROJECT}/results/p1_m11"
ARTIFACT_DIR="${BASE}/artifacts"

TS="$(date '+%Y%m%d_%H%M%S')"

mkdir -p \
    "${SRC_DIR}" \
    "${RESULT_DIR}" \
    "${ARTIFACT_DIR}"

TXT="${ARTIFACT_DIR}/P1_M11_NCU_KERNEL_PROFILING_${TS}.txt"

CUDA_SRC="${SRC_DIR}/p1_m11_llm_kernel_proxy.cu"
CUDA_BIN="${SRC_DIR}/p1_m11_llm_kernel_proxy"

SMOKE_CSV="${RESULT_DIR}/P1_M11_SMOKE_${TS}.csv"
PROFILE_CSV="${RESULT_DIR}/P1_M11_PROFILE_${TS}.csv"

NCU_RAW="${RESULT_DIR}/P1_M11_NCU_RAW_${TS}.txt"
NCU_REP="${RESULT_DIR}/P1_M11_NCU_${TS}.ncu-rep"

LINE="========================================================================================================================"


main() {
    set -e

    export CUDA_HOME="/usr/local/cuda-12.8"
    export PATH="${CUDA_HOME}/bin:${ENV}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

    echo "${LINE}"
    echo "AI INFRA PROJECT 1 - P1-M11"
    echo "NSIGHT COMPUTE KERNEL-LEVEL PROFILING"
    echo "${LINE}"

    echo
    echo "TIMESTAMP = $(date '+%Y-%m-%d %H:%M:%S %z')"


    # =================================================================================================
    # 1. ENVIRONMENT
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "1. ENVIRONMENT"
    echo "${LINE}"

    echo "NCU = $(command -v ncu || echo NOT_FOUND)"
    echo "NVCC = $(command -v nvcc || echo NOT_FOUND)"

    if ! command -v ncu >/dev/null 2>&1; then
        echo "ERROR = NSIGHT_COMPUTE_NOT_FOUND"
        return 2
    fi

    if ! command -v nvcc >/dev/null 2>&1; then
        echo "ERROR = NVCC_NOT_FOUND"
        return 2
    fi

    ncu --version | head -n 10

    echo
    nvcc --version | tail -n 5

    echo
    nvidia-smi \
        --query-gpu=name,driver_version,memory.total,compute_cap \
        --format=csv,noheader


    # =================================================================================================
    # 2. NCU PERMISSION SMOKE TEST
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "2. NCU PROFILING PERMISSION SMOKE TEST"
    echo "${LINE}"

    P1_M3_BIN="${PROJECT}/src/p1_m3/cuda_memory_warp"

    if [ ! -x "${P1_M3_BIN}" ]; then
        echo "P1_M3_BINARY = NOT_AVAILABLE"
        echo "SMOKE_TEST_SOURCE = P1_M11_PROXY_LATER"
        SMOKE_AVAILABLE="FALSE"
    else
        echo "P1_M3_BINARY = ${P1_M3_BIN}"
        SMOKE_AVAILABLE="TRUE"

        set +e

        ncu \
            --csv \
            --page raw \
            --launch-count 1 \
            --metrics \
sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
            "${P1_M3_BIN}" \
            > "${SMOKE_CSV}" 2>&1

        SMOKE_STATUS=$?

        set -e

        echo "NCU_SMOKE_EXIT_CODE = ${SMOKE_STATUS}"

        cat "${SMOKE_CSV}" | head -n 80

        if grep -Eqi \
            'ERR_NVGPUCTRPERM|permission|not permitted|access.*performance counter' \
            "${SMOKE_CSV}"
        then
            echo "NCU_HW_COUNTER_PERMISSION = DENIED"
            NCU_PERMISSION="DENIED"

        elif [ "${SMOKE_STATUS}" -eq 0 ]; then
            echo "NCU_HW_COUNTER_PERMISSION = AVAILABLE"
            NCU_PERMISSION="AVAILABLE"

        else
            echo "NCU_HW_COUNTER_PERMISSION = UNKNOWN"
            NCU_PERMISSION="UNKNOWN"
        fi
    fi


    # =================================================================================================
    # 3. CREATE LLM-LIKE CUDA KERNEL WORKLOAD
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "3. CREATE LLM-LIKE CUDA KERNEL WORKLOAD"
    echo "${LINE}"

    cat > "${CUDA_SRC}" <<'CU'
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
CU

    echo "CUDA_SOURCE = ${CUDA_SRC}"


    # =================================================================================================
    # 4. COMPILE
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "4. COMPILE"
    echo "${LINE}"

    nvcc \
        -O3 \
        -lineinfo \
        -arch=sm_89 \
        "${CUDA_SRC}" \
        -o "${CUDA_BIN}"

    echo "CUDA_COMPILE = SUCCESS"
    echo "CUDA_BINARY = ${CUDA_BIN}"


    # =================================================================================================
    # 5. NATIVE EXECUTION
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "5. NATIVE EXECUTION"
    echo "${LINE}"

    "${CUDA_BIN}"

    echo "NATIVE_EXECUTION = SUCCESS"


    # =================================================================================================
    # 6. NCU PROFILE
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "6. NSIGHT COMPUTE PROFILE"
    echo "${LINE}"

    METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__occupancy_limit_blocks"

    set +e

    ncu \
        --force-overwrite \
        --target-processes all \
        --kernel-name-base demangled \
        --launch-skip 4 \
        --launch-count 12 \
        --metrics "${METRICS}" \
        --export "${NCU_REP}" \
        "${CUDA_BIN}" \
        > "${NCU_RAW}" 2>&1

    NCU_STATUS=$?

    set -e

    echo "NCU_EXIT_CODE = ${NCU_STATUS}"

    echo
    echo "NCU_RAW_OUTPUT:"
    cat "${NCU_RAW}"


    # =================================================================================================
    # 7. HANDLE CLOUD PROFILING PERMISSION
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "7. NCU PERMISSION / COLLECTION VERDICT"
    echo "${LINE}"

    if grep -Eqi \
        'ERR_NVGPUCTRPERM|not permitted|performance counter.*permission|permission.*counter' \
        "${NCU_RAW}"
    then

        echo "NCU_HARDWARE_COUNTERS = BLOCKED_BY_HOST"

        echo
        echo "INTERPRETATION:"
        echo "The CUDA program itself runs correctly."
        echo "The cloud host/container does not expose the NVIDIA"
        echo "hardware performance counters required by Nsight Compute."
        echo "This is an infrastructure permission limitation,"
        echo "not a CUDA kernel failure."

        echo
        echo "VERDICT = NCU_HOST_PERMISSION_BLOCKED"

        return 30
    fi


    if [ "${NCU_STATUS}" -ne 0 ]; then

        echo "NCU_HARDWARE_COUNTERS = UNKNOWN_FAILURE"
        echo "VERDICT = P1_M11_NCU_REQUIRES_DIAGNOSIS"

        return 31
    fi


    echo "NCU_HARDWARE_COUNTERS = AVAILABLE"


    # =================================================================================================
    # 8. EXPORT STRUCTURED NCU DATA
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "8. EXPORT STRUCTURED NCU DATA"
    echo "${LINE}"

    ncu \
        --import "${NCU_REP}" \
        --csv \
        --page raw \
        > "${PROFILE_CSV}" \
        2>&1

    echo "NCU_CSV = ${PROFILE_CSV}"


    # =================================================================================================
    # 9. PARSE IMPORTANT METRICS
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "9. KERNEL METRIC SUMMARY"
    echo "${LINE}"

    "${PYTHON}" - \
        "${PROFILE_CSV}" \
        <<'PY'
import csv
import io
import re
import sys
from collections import defaultdict


path = sys.argv[1]

text = open(
    path,
    "r",
    encoding="utf-8",
    errors="replace",
).read()


print(
    "PROFILE_FILE =",
    path,
)


names = [
    "kv_copy_kernel",
    "rmsnorm_kernel",
    "softmax_kernel",
    "tiled_gemm_kernel",
]


for name in names:

    lines = [
        line
        for line in text.splitlines()
        if name in line
    ]

    print()
    print("=" * 100)

    print(
        "KERNEL =",
        name,
    )

    print(
        "MATCHED_ROWS =",
        len(lines),
    )

    for line in lines[:30]:
        print(line)


print()
print("=" * 100)
print("KERNEL-LEVEL INTERPRETATION")
print("=" * 100)

print(
"""
KV COPY

Expected character:
    low arithmetic intensity
    memory-oriented data movement

Important evidence:
    DRAM throughput
    SM throughput


RMSNORM

Expected character:
    reductions + elementwise arithmetic
    memory traffic plus synchronization

Important evidence:
    SM throughput
    DRAM throughput
    occupancy


SOFTMAX

Expected character:
    reductions
    exponentials
    synchronization
    repeated memory access

Important evidence:
    duration
    SM throughput
    occupancy


TILED GEMM

Expected character:
    much higher arithmetic intensity
    shared-memory reuse
    compute-oriented execution

Important evidence:
    SM throughput relative to DRAM activity


The purpose of this experiment is NOT to claim these
hand-written kernels reproduce production vLLM kernels.

They are controlled conformance cases establishing the
relationship between:

    kernel structure
    arithmetic intensity
    memory behavior
    occupancy
    achieved throughput

The next stage can apply the same profiling methodology
to production inference kernels.
""".strip()
)
PY


    # =================================================================================================
    # 10. KNOWLEDGE CHECKPOINT
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "10. P1-M11 KNOWLEDGE CHECKPOINT"
    echo "${LINE}"

    echo "
NSIGHT COMPUTE CONNECTS SOURCE-LEVEL KERNEL DESIGN TO
GPU HARDWARE COUNTERS.

APPLICATION METRICS:

    TTFT
    TPOT
    tokens/s

do not explain WHY a GPU kernel behaves as observed.

NCU adds hardware evidence such as:

    kernel duration
    SM throughput
    DRAM throughput
    active warps
    occupancy constraints

Together the debugging stack becomes:

    Serving metric
        ↓
    Scheduler / KV metric
        ↓
    GPU telemetry
        ↓
    CUDA kernel
        ↓
    Hardware counter

This is the transition from ordinary LLM application
benchmarking to GPU systems performance engineering.
"


    # =================================================================================================
    # 11. FINAL
    # =================================================================================================

    echo
    echo "${LINE}"
    echo "FINAL VERDICT"
    echo "${LINE}"

    echo "CUDA_KERNEL_SUITE = SUCCESS"
    echo "NSIGHT_COMPUTE = SUCCESS"
    echo "HARDWARE_COUNTER_COLLECTION = SUCCESS"
    echo "KERNEL_LEVEL_PROFILE = SUCCESS"
    echo "STRUCTURED_NCU_REPORT = SUCCESS"

    echo "VERDICT = READY_FOR_P1_M12_VLLM_PRODUCTION_KERNEL_PROFILING"

    echo
    echo "${LINE}"
    echo "ARTIFACTS"
    echo "${LINE}"

    echo "CUDA_SOURCE = ${CUDA_SRC}"
    echo "CUDA_BINARY = ${CUDA_BIN}"
    echo "NCU_REPORT = ${NCU_REP}"
    echo "NCU_CSV = ${PROFILE_CSV}"
    echo "NCU_RAW = ${NCU_RAW}"

    echo
    echo "${LINE}"
    echo "END OF P1-M11"
    echo "${LINE}"
}


set +e

main 2>&1 | tee "${TXT}"

STATUS=${PIPESTATUS[0]}

set -e

printf '%s\n' "${TXT}"

exit "${STATUS}"
