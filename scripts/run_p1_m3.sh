#!/usr/bin/env bash
set -uo pipefail

BASE="/root/autodl-tmp/ai-infra"
PROJECT="${BASE}/project1-turbollm"
SRC_DIR="${PROJECT}/src/p1_m3"
RESULT_DIR="${PROJECT}/results/p1_m3"
ARTIFACT_DIR="${BASE}/artifacts"

TS="$(date '+%Y%m%d_%H%M%S')"

mkdir -p \
    "${SRC_DIR}" \
    "${RESULT_DIR}" \
    "${ARTIFACT_DIR}"

TXT="${ARTIFACT_DIR}/P1_M3_CUDA_MEMORY_WARP_${TS}.txt"
CSV="${RESULT_DIR}/P1_M3_RESULTS_${TS}.csv"

CU="${SRC_DIR}/cuda_memory_warp.cu"
BIN="${SRC_DIR}/cuda_memory_warp"

CUDA_HOME="/usr/local/cuda-12.8"
NVCC="${CUDA_HOME}/bin/nvcc"

# Preserve original terminal stdout.
exec 3>&1

# Everything from here goes into TXT and terminal simultaneously.
exec > >(tee "${TXT}") 2>&1

finish() {
    STATUS=$?

    trap - EXIT

    echo
    echo "========================================================================================================================"
    echo "SCRIPT COMPLETION"
    echo "========================================================================================================================"
    echo "EXIT_CODE = ${STATUS}"
    echo "TXT_RESULT = ${TXT}"

    if [ "${STATUS}" -eq 0 ]; then
        echo "SCRIPT_STATUS = SUCCESS"
    else
        echo "SCRIPT_STATUS = FAILED"
    fi

    # Restore original terminal output.
    exec 1>&3 2>&3

    # The final terminal line is ONLY the absolute TXT path.
    printf '%s\n' "${TXT}"

    exec 3>&-

    exit "${STATUS}"
}

trap finish EXIT

set -e

LINE="========================================================================================================================"

echo "${LINE}"
echo "AI INFRA PROJECT 1 - P1-M3"
echo "CUDA MEMORY ACCESS AND WARP BEHAVIOR"
echo "${LINE}"

echo
echo "TIMESTAMP = $(date '+%Y-%m-%d %H:%M:%S %z')"

echo
echo "${LINE}"
echo "1. ENVIRONMENT"
echo "${LINE}"

echo "CUDA_HOME = ${CUDA_HOME}"
echo "NVCC = ${NVCC}"

if [ ! -x "${NVCC}" ]; then
    echo "NVCC_STATUS = NOT_FOUND"
    exit 1
fi

"${NVCC}" --version

echo
nvidia-smi \
    --query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,power.limit \
    --format=csv,noheader

echo
echo "${LINE}"
echo "2. CREATE CUDA SOURCE"
echo "${LINE}"

cat > "${CU}" <<'CPP'
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
do {                                                                           \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
        std::cerr                                                              \
            << "CUDA ERROR: "                                                  \
            << cudaGetErrorString(err__)                                       \
            << " at "                                                          \
            << __FILE__                                                        \
            << ":"                                                             \
            << __LINE__                                                        \
            << std::endl;                                                      \
        std::exit(1);                                                          \
    }                                                                          \
} while (0)


constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;


__global__ void init_kernel(
    float* data,
    size_t n
) {
    size_t gid =
        static_cast<size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (gid < n) {
        data[gid] =
            static_cast<float>(gid % 1024) * 0.001f;
    }
}


/*
 * Neighboring threads access neighboring addresses.
 *
 * Warp:
 *
 * thread 0 -> input[i]
 * thread 1 -> input[i+1]
 * thread 2 -> input[i+2]
 *
 * This is a coalesced global-memory access pattern.
 */
__global__ void coalesced_copy_kernel(
    const float* input,
    float* output,
    int width,
    int height
) {
    int x =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    int y =
        blockIdx.y * blockDim.y
        + threadIdx.y;

    if (x < width && y < height) {
        size_t idx =
            static_cast<size_t>(y) * width
            + x;

        output[idx] =
            input[idx];
    }
}


/*
 * Neighboring threads read addresses separated by "height".
 *
 * thread 0 -> input[y]
 * thread 1 -> input[height + y]
 * thread 2 -> input[2*height + y]
 *
 * This intentionally creates a strided / poorly-coalesced read pattern.
 */
__global__ void strided_read_kernel(
    const float* input,
    float* output,
    int width,
    int height
) {
    int x =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    int y =
        blockIdx.y * blockDim.y
        + threadIdx.y;

    if (x < width && y < height) {
        size_t output_idx =
            static_cast<size_t>(y) * width
            + x;

        size_t input_idx =
            static_cast<size_t>(x) * height
            + y;

        output[output_idx] =
            input[input_idx];
    }
}


/*
 * Naive matrix transpose.
 *
 * Input reads are coalesced.
 * Output writes are strided.
 */
__global__ void transpose_naive_kernel(
    float* output,
    const float* input,
    int width,
    int height
) {
    int x =
        blockIdx.x * TILE_DIM
        + threadIdx.x;

    int y =
        blockIdx.y * TILE_DIM
        + threadIdx.y;

    for (
        int j = 0;
        j < TILE_DIM;
        j += BLOCK_ROWS
    ) {
        if (
            x < width
            &&
            y + j < height
        ) {
            output[
                static_cast<size_t>(x) * height
                + (y + j)
            ] =
                input[
                    static_cast<size_t>(y + j) * width
                    + x
                ];
        }
    }
}


/*
 * Shared-memory tiled transpose.
 *
 * Phase 1:
 *   global memory -> shared memory
 *   coalesced read
 *
 * Phase 2:
 *   shared memory -> global memory
 *   coalesced write
 *
 * TILE_DIM + 1 avoids the classic 32-way shared-memory
 * bank-conflict problem during transpose.
 */
__global__ void transpose_tiled_kernel(
    float* output,
    const float* input,
    int width,
    int height
) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x =
        blockIdx.x * TILE_DIM
        + threadIdx.x;

    int y =
        blockIdx.y * TILE_DIM
        + threadIdx.y;

    for (
        int j = 0;
        j < TILE_DIM;
        j += BLOCK_ROWS
    ) {
        if (
            x < width
            &&
            y + j < height
        ) {
            tile[
                threadIdx.y + j
            ][
                threadIdx.x
            ] =
                input[
                    static_cast<size_t>(y + j) * width
                    + x
                ];
        }
    }

    __syncthreads();

    x =
        blockIdx.y * TILE_DIM
        + threadIdx.x;

    y =
        blockIdx.x * TILE_DIM
        + threadIdx.y;

    for (
        int j = 0;
        j < TILE_DIM;
        j += BLOCK_ROWS
    ) {
        if (
            x < height
            &&
            y + j < width
        ) {
            output[
                static_cast<size_t>(y + j) * height
                + x
            ] =
                tile[
                    threadIdx.x
                ][
                    threadIdx.y + j
                ];
        }
    }
}


__device__ __noinline__
float branch_path_a(float x) {
    #pragma unroll 1
    for (int i = 0; i < 64; ++i) {
        x = fmaf(
            x,
            1.000001f,
            0.000001f
        );
    }

    return x;
}


__device__ __noinline__
float branch_path_b(float x) {
    #pragma unroll 1
    for (int i = 0; i < 64; ++i) {
        x = fmaf(
            x,
            0.999999f,
            -0.000001f
        );
    }

    return x;
}


/*
 * All 32 lanes in each warp take the SAME branch.
 *
 * Different warps may choose different paths,
 * but there is no within-warp divergence.
 */
__global__ void uniform_branch_kernel(
    const float* input,
    float* output,
    size_t n
) {
    size_t gid =
        static_cast<size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (gid >= n) {
        return;
    }

    size_t warp_id =
        gid >> 5;

    float x =
        input[gid];

    if ((warp_id & 1ULL) == 0) {
        output[gid] =
            branch_path_a(x);
    } else {
        output[gid] =
            branch_path_b(x);
    }
}


/*
 * Half of each warp takes path A.
 * Half takes path B.
 *
 * This intentionally creates warp divergence.
 */
__global__ void divergent_branch_kernel(
    const float* input,
    float* output,
    size_t n
) {
    size_t gid =
        static_cast<size_t>(blockIdx.x) * blockDim.x
        + threadIdx.x;

    if (gid >= n) {
        return;
    }

    int lane =
        threadIdx.x & 31;

    float x =
        input[gid];

    if (lane < 16) {
        output[gid] =
            branch_path_a(x);
    } else {
        output[gid] =
            branch_path_b(x);
    }
}


struct Result {
    std::string experiment;
    std::string variant;
    double time_ms;
    double useful_gib_s;
};


template <typename LaunchFn>
double benchmark_kernel(
    LaunchFn launch,
    int warmup,
    int repeats
) {
    for (int i = 0; i < warmup; ++i) {
        launch();
    }

    CUDA_CHECK(
        cudaGetLastError()
    );

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(
        cudaEventCreate(&start)
    );

    CUDA_CHECK(
        cudaEventCreate(&stop)
    );

    CUDA_CHECK(
        cudaEventRecord(start)
    );

    for (int i = 0; i < repeats; ++i) {
        launch();
    }

    CUDA_CHECK(
        cudaEventRecord(stop)
    );

    CUDA_CHECK(
        cudaEventSynchronize(stop)
    );

    float total_ms = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        )
    );

    CUDA_CHECK(
        cudaEventDestroy(start)
    );

    CUDA_CHECK(
        cudaEventDestroy(stop)
    );

    return
        static_cast<double>(total_ms)
        / repeats;
}


float read_device_value(
    const float* ptr,
    size_t index
) {
    float value = 0.0f;

    CUDA_CHECK(
        cudaMemcpy(
            &value,
            ptr + index,
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    return value;
}


float expected_input_value(
    size_t index
) {
    return
        static_cast<float>(
            index % 1024
        )
        * 0.001f;
}


void print_device_info() {
    cudaDeviceProp prop{};

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            0
        )
    );

    std::cout
        << "GPU_NAME = "
        << prop.name
        << "\n";

    std::cout
        << "COMPUTE_CAPABILITY = "
        << prop.major
        << "."
        << prop.minor
        << "\n";

    std::cout
        << "SM_COUNT = "
        << prop.multiProcessorCount
        << "\n";

    std::cout
        << "WARP_SIZE = "
        << prop.warpSize
        << "\n";

    std::cout
        << "MAX_THREADS_PER_BLOCK = "
        << prop.maxThreadsPerBlock
        << "\n";

    std::cout
        << "MAX_THREADS_PER_SM = "
        << prop.maxThreadsPerMultiProcessor
        << "\n";

    std::cout
        << "SHARED_MEMORY_PER_BLOCK_BYTES = "
        << prop.sharedMemPerBlock
        << "\n";

    std::cout
        << "SHARED_MEMORY_PER_SM_BYTES = "
        << prop.sharedMemPerMultiprocessor
        << "\n";

    std::cout
        << "REGISTERS_PER_BLOCK = "
        << prop.regsPerBlock
        << "\n";

    std::cout
        << "L2_CACHE_BYTES = "
        << prop.l2CacheSize
        << "\n";

    std::cout
        << "GLOBAL_MEMORY_GIB = "
        << std::fixed
        << std::setprecision(3)
        << static_cast<double>(
            prop.totalGlobalMem
        )
        / (
            1024.0
            * 1024.0
            * 1024.0
        )
        << "\n";
}


int main(
    int argc,
    char** argv
) {
    if (argc != 2) {
        std::cerr
            << "Usage: "
            << argv[0]
            << " result.csv\n";

        return 1;
    }

    const std::string csv_path =
        argv[1];

    const std::string LINE(
        120,
        '='
    );

    std::vector<Result> results;


    std::cout
        << LINE
        << "\n"
        << "CUDA DEVICE CHARACTERIZATION"
        << "\n"
        << LINE
        << "\n";

    print_device_info();


    /*
     * 8192 x 8192
     *
     * 67,108,864 float elements
     * 256 MiB per array
     */
    constexpr int WIDTH = 8192;
    constexpr int HEIGHT = 8192;

    constexpr size_t N =
        static_cast<size_t>(WIDTH)
        * HEIGHT;

    const size_t BYTES =
        N * sizeof(float);


    float* input = nullptr;
    float* output = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &input,
            BYTES
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &output,
            BYTES
        )
    );


    constexpr int INIT_BLOCK = 256;

    const int init_grid =
        static_cast<int>(
            (N + INIT_BLOCK - 1)
            / INIT_BLOCK
        );

    init_kernel<<<
        init_grid,
        INIT_BLOCK
    >>>(
        input,
        N
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    /*
     * =====================================================================
     * EXPERIMENT 1
     * =====================================================================
     */

    std::cout
        << "\n"
        << LINE
        << "\n"
        << "EXPERIMENT 1 — GLOBAL MEMORY COALESCING"
        << "\n"
        << LINE
        << "\n";

    std::cout
        << "MATRIX_WIDTH = "
        << WIDTH
        << "\n";

    std::cout
        << "MATRIX_HEIGHT = "
        << HEIGHT
        << "\n";

    std::cout
        << "ELEMENT_COUNT = "
        << N
        << "\n";

    std::cout
        << "ARRAY_SIZE_MIB = "
        << std::fixed
        << std::setprecision(3)
        << static_cast<double>(BYTES)
           / (1024.0 * 1024.0)
        << "\n";


    dim3 memory_block(
        32,
        8
    );

    dim3 memory_grid(
        (WIDTH + memory_block.x - 1)
            / memory_block.x,

        (HEIGHT + memory_block.y - 1)
            / memory_block.y
    );


    double coalesced_ms =
        benchmark_kernel(
            [&]() {
                coalesced_copy_kernel<<<
                    memory_grid,
                    memory_block
                >>>(
                    input,
                    output,
                    WIDTH,
                    HEIGHT
                );
            },
            10,
            50
        );


    double strided_ms =
        benchmark_kernel(
            [&]() {
                strided_read_kernel<<<
                    memory_grid,
                    memory_block
                >>>(
                    input,
                    output,
                    WIDTH,
                    HEIGHT
                );
            },
            10,
            50
        );


    /*
     * One read + one write:
     * 8 useful bytes per element.
     */
    const double useful_bytes =
        static_cast<double>(N)
        * 2.0
        * sizeof(float);

    const double useful_gib =
        useful_bytes
        / (
            1024.0
            * 1024.0
            * 1024.0
        );


    double coalesced_bw =
        useful_gib
        / (coalesced_ms / 1000.0);

    double strided_bw =
        useful_gib
        / (strided_ms / 1000.0);


    std::cout
        << "\n"
        << "COALESCED_TIME_MS = "
        << std::fixed
        << std::setprecision(6)
        << coalesced_ms
        << "\n";

    std::cout
        << "COALESCED_USEFUL_GIB_S = "
        << std::setprecision(3)
        << coalesced_bw
        << "\n";

    std::cout
        << "\n"
        << "STRIDED_TIME_MS = "
        << std::setprecision(6)
        << strided_ms
        << "\n";

    std::cout
        << "STRIDED_USEFUL_GIB_S = "
        << std::setprecision(3)
        << strided_bw
        << "\n";

    std::cout
        << "\n"
        << "STRIDED_SLOWDOWN = "
        << std::setprecision(3)
        << strided_ms / coalesced_ms
        << "x\n";


    results.push_back({
        "global_memory",
        "coalesced",
        coalesced_ms,
        coalesced_bw
    });

    results.push_back({
        "global_memory",
        "strided",
        strided_ms,
        strided_bw
    });


    /*
     * =====================================================================
     * EXPERIMENT 2
     * =====================================================================
     */

    std::cout
        << "\n"
        << LINE
        << "\n"
        << "EXPERIMENT 2 — SHARED MEMORY TILED TRANSPOSE"
        << "\n"
        << LINE
        << "\n";


    dim3 transpose_block(
        TILE_DIM,
        BLOCK_ROWS
    );

    dim3 transpose_grid(
        (WIDTH + TILE_DIM - 1)
            / TILE_DIM,

        (HEIGHT + TILE_DIM - 1)
            / TILE_DIM
    );


    double naive_ms =
        benchmark_kernel(
            [&]() {
                transpose_naive_kernel<<<
                    transpose_grid,
                    transpose_block
                >>>(
                    output,
                    input,
                    WIDTH,
                    HEIGHT
                );
            },
            10,
            50
        );


    double tiled_ms =
        benchmark_kernel(
            [&]() {
                transpose_tiled_kernel<<<
                    transpose_grid,
                    transpose_block
                >>>(
                    output,
                    input,
                    WIDTH,
                    HEIGHT
                );
            },
            10,
            50
        );


    double naive_bw =
        useful_gib
        / (naive_ms / 1000.0);

    double tiled_bw =
        useful_gib
        / (tiled_ms / 1000.0);


    std::cout
        << "NAIVE_TRANSPOSE_MS = "
        << std::fixed
        << std::setprecision(6)
        << naive_ms
        << "\n";

    std::cout
        << "NAIVE_USEFUL_GIB_S = "
        << std::setprecision(3)
        << naive_bw
        << "\n";

    std::cout
        << "\n"
        << "TILED_TRANSPOSE_MS = "
        << std::setprecision(6)
        << tiled_ms
        << "\n";

    std::cout
        << "TILED_USEFUL_GIB_S = "
        << std::setprecision(3)
        << tiled_bw
        << "\n";

    std::cout
        << "\n"
        << "TILED_SPEEDUP = "
        << std::setprecision(3)
        << naive_ms / tiled_ms
        << "x\n";


    results.push_back({
        "transpose",
        "naive",
        naive_ms,
        naive_bw
    });

    results.push_back({
        "transpose",
        "shared_memory_tiled",
        tiled_ms,
        tiled_bw
    });


    /*
     * Validate several transposed locations.
     */
    std::vector<std::pair<int, int>> samples = {
        {0, 0},
        {1, 2},
        {31, 37},
        {1024, 2048},
        {4095, 4096},
        {8191, 8191}
    };

    bool transpose_valid = true;

    for (const auto& sample : samples) {
        int row = sample.first;
        int col = sample.second;

        size_t source_index =
            static_cast<size_t>(row)
            * WIDTH
            + col;

        size_t output_index =
            static_cast<size_t>(col)
            * HEIGHT
            + row;

        float observed =
            read_device_value(
                output,
                output_index
            );

        float expected =
            expected_input_value(
                source_index
            );

        double diff =
            std::fabs(
                static_cast<double>(observed)
                - expected
            );

        std::cout
            << "TRANSPOSE_SAMPLE"
            << " row="
            << row
            << " col="
            << col
            << " expected="
            << expected
            << " observed="
            << observed
            << " diff="
            << diff
            << "\n";

        if (diff > 1e-6) {
            transpose_valid = false;
        }
    }

    if (!transpose_valid) {
        std::cerr
            << "TRANSPOSE_VALIDATION = FAILED\n";

        return 2;
    }

    std::cout
        << "TRANSPOSE_VALIDATION = SUCCESS\n";


    /*
     * =====================================================================
     * EXPERIMENT 3
     * =====================================================================
     */

    std::cout
        << "\n"
        << LINE
        << "\n"
        << "EXPERIMENT 3 — WARP DIVERGENCE"
        << "\n"
        << LINE
        << "\n";


    constexpr size_t BRANCH_N =
        1ULL << 24;

    constexpr int BRANCH_BLOCK =
        256;

    const int branch_grid =
        static_cast<int>(
            (
                BRANCH_N
                + BRANCH_BLOCK
                - 1
            )
            / BRANCH_BLOCK
        );


    double uniform_ms =
        benchmark_kernel(
            [&]() {
                uniform_branch_kernel<<<
                    branch_grid,
                    BRANCH_BLOCK
                >>>(
                    input,
                    output,
                    BRANCH_N
                );
            },
            5,
            20
        );


    double divergent_ms =
        benchmark_kernel(
            [&]() {
                divergent_branch_kernel<<<
                    branch_grid,
                    BRANCH_BLOCK
                >>>(
                    input,
                    output,
                    BRANCH_N
                );
            },
            5,
            20
        );


    std::cout
        << "BRANCH_ELEMENTS = "
        << BRANCH_N
        << "\n";

    std::cout
        << "BLOCK_SIZE = "
        << BRANCH_BLOCK
        << "\n";

    std::cout
        << "WARPS_PER_BLOCK = "
        << BRANCH_BLOCK / 32
        << "\n";

    std::cout
        << "\n"
        << "UNIFORM_BRANCH_MS = "
        << std::fixed
        << std::setprecision(6)
        << uniform_ms
        << "\n";

    std::cout
        << "DIVERGENT_BRANCH_MS = "
        << divergent_ms
        << "\n";

    std::cout
        << "DIVERGENCE_SLOWDOWN = "
        << std::setprecision(3)
        << divergent_ms / uniform_ms
        << "x\n";


    results.push_back({
        "warp_branch",
        "uniform",
        uniform_ms,
        0.0
    });

    results.push_back({
        "warp_branch",
        "divergent",
        divergent_ms,
        0.0
    });


    /*
     * =====================================================================
     * SAVE CSV
     * =====================================================================
     */

    std::ofstream csv(
        csv_path
    );

    if (!csv.is_open()) {
        std::cerr
            << "CSV_OPEN = FAILED\n";

        return 3;
    }


    csv
        << "experiment,"
        << "variant,"
        << "time_ms,"
        << "useful_gib_s\n";


    for (const auto& result : results) {
        csv
            << result.experiment
            << ","
            << result.variant
            << ","
            << std::fixed
            << std::setprecision(6)
            << result.time_ms
            << ","
            << std::setprecision(3)
            << result.useful_gib_s
            << "\n";
    }

    csv.close();


    CUDA_CHECK(
        cudaFree(input)
    );

    CUDA_CHECK(
        cudaFree(output)
    );


    std::cout
        << "\n"
        << LINE
        << "\n"
        << "P1-M3 KNOWLEDGE CHECKPOINT"
        << "\n"
        << LINE
        << "\n";


    std::cout << R"(
CUDA MEMORY HIERARCHY — simplified:

CPU RAM
   │
   │ PCIe
   ↓
GPU Global Memory / VRAM
   │
   ↓
L2 Cache
   │
   ↓
SM
 ├── L1
 ├── Shared Memory
 └── Registers


KEY IDEA 1 — MEMORY COALESCING

A warp contains 32 threads.

Best case:

thread 0  -> address N
thread 1  -> address N+1
thread 2  -> address N+2
...
thread 31 -> address N+31

These neighboring accesses can be combined into efficient
global-memory transactions.

Poorly-coalesced / strided accesses require more memory
transactions and can reduce effective memory bandwidth.


KEY IDEA 2 — SHARED MEMORY

Shared memory:

- lives on the SM
- has much lower latency than global memory
- is shared by threads in the same block
- can be used to reorganize memory access

Matrix transpose is a classic example.

Naive transpose:

coalesced read
+
strided write

Tiled transpose:

coalesced global read
→ shared-memory tile
→ coalesced global write


KEY IDEA 3 — SHARED-MEMORY BANK CONFLICTS

Shared memory is divided into banks.

The tile is declared as:

    float tile[32][33]

instead of:

    float tile[32][32]

The +1 padding changes address mapping and avoids the
classic 32-way bank conflict during transpose.


KEY IDEA 4 — WARP DIVERGENCE

Threads are scheduled in warps of 32.

If all threads in a warp execute the same branch:

    efficient SIMT execution

If half execute branch A and half execute branch B:

    the warp may need to execute both paths
    with different lanes masked

This is warp divergence.


WHY THIS MATTERS FOR AI INFRA

Later LLM workloads involve:

- Attention
- KV Cache access
- RMSNorm
- Softmax
- Quantization kernels
- Sampling
- MoE routing

Their performance depends heavily on:

    memory access pattern
    memory bandwidth
    shared-memory usage
    synchronization
    warp behavior
    arithmetic intensity

These are fundamental GPU-performance concepts.
)" << std::endl;


    std::cout
        << LINE
        << "\n"
        << "FINAL VERDICT"
        << "\n"
        << LINE
        << "\n";

    std::cout
        << "CUDA_KERNEL_EXECUTION = SUCCESS\n"
        << "GLOBAL_MEMORY_COALESCING_TEST = SUCCESS\n"
        << "SHARED_MEMORY_TRANSPOSE_TEST = SUCCESS\n"
        << "TRANSPOSE_VALIDATION = SUCCESS\n"
        << "WARP_DIVERGENCE_TEST = SUCCESS\n"
        << "STRUCTURED_CSV_RESULT = SUCCESS\n"
        << "VERDICT = READY_FOR_P1_M4_GPU_PERFORMANCE_MODEL\n";

    return 0;
}
CPP

echo "CUDA_SOURCE_CREATED = ${CU}"

echo
echo "${LINE}"
echo "3. COMPILE"
echo "${LINE}"

"${NVCC}" \
    -O3 \
    -lineinfo \
    -arch=sm_89 \
    "${CU}" \
    -o "${BIN}"

echo "CUDA_COMPILE = SUCCESS"
echo "BINARY = ${BIN}"

echo
echo "${LINE}"
echo "4. RUN EXPERIMENTS"
echo "${LINE}"

"${BIN}" "${CSV}"

echo
echo "${LINE}"
echo "5. STRUCTURED RESULT FILE"
echo "${LINE}"

echo "CSV = ${CSV}"

echo
echo "${LINE}"
echo "6. CSV CONTENT INCLUDED IN TXT"
echo "${LINE}"

cat "${CSV}"

echo
echo "${LINE}"
echo "7. OUTPUT ARTIFACTS"
echo "${LINE}"

echo "CUDA_SOURCE = ${CU}"
echo "CUDA_BINARY = ${BIN}"
echo "CSV_RESULT = ${CSV}"
echo "TXT_RESULT = ${TXT}"

echo
echo "${LINE}"
echo "END OF P1-M3 RESULT"
echo "${LINE}"
