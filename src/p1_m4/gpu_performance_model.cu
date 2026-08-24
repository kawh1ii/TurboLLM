#include <cuda_runtime.h>

#include <algorithm>
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


constexpr int BLOCK_SIZE = 256;


__global__ void init_kernel(
    float* input,
    size_t n
) {
    size_t gid =
        static_cast<size_t>(blockIdx.x)
        * blockDim.x
        + threadIdx.x;

    if (gid < n) {
        input[gid] =
            0.5f
            +
            static_cast<float>(
                gid % 1024
            )
            * 0.0001f;
    }
}


__global__ void streaming_copy_kernel(
    const float* input,
    float* output,
    size_t n
) {
    size_t gid =
        static_cast<size_t>(blockIdx.x)
        * blockDim.x
        + threadIdx.x;

    if (gid < n) {
        output[gid] =
            input[gid];
    }
}


/*
 * Each loop iteration executes four FP32 FMAs.
 *
 * One FMA = multiply + add = 2 FLOPs.
 *
 * Therefore:
 *
 *   4 FMAs × 2 FLOPs = 8 FLOPs / iteration
 *
 * Global-memory traffic per element:
 *
 *   load input  = 4 bytes
 *   store output = 4 bytes
 *
 * Total useful traffic ≈ 8 bytes.
 *
 * Arithmetic intensity:
 *
 *   8 * ITERS FLOPs / 8 bytes
 *   = ITERS FLOP/byte
 *
 * Increasing ITERS moves the workload from
 * memory-dominated toward compute-dominated.
 */
template<int ITERS>
__global__ void arithmetic_intensity_kernel(
    const float* input,
    float* output,
    size_t n
) {
    size_t gid =
        static_cast<size_t>(blockIdx.x)
        * blockDim.x
        + threadIdx.x;

    if (gid >= n) {
        return;
    }

    float x =
        input[gid];

    float a =
        x + 0.001f;

    float b =
        x + 0.002f;

    float c =
        x + 0.003f;

    float d =
        x + 0.004f;

    #pragma unroll 4
    for (int i = 0; i < ITERS; ++i) {
        a = fmaf(
            a,
            1.000001f,
            0.000001f
        );

        b = fmaf(
            b,
            0.999999f,
            0.000002f
        );

        c = fmaf(
            c,
            1.000002f,
            -0.000001f
        );

        d = fmaf(
            d,
            0.999998f,
            0.000003f
        );
    }

    output[gid] =
        a + b + c + d;
}


struct DeviceInfo {
    int sm_count;
    int warp_size;
    int max_threads_per_sm;

    int memory_bus_width_bits;

    int memory_clock_khz;

    double theoretical_memory_gb_s;
};


DeviceInfo print_device_info() {
    cudaDeviceProp prop{};

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            0
        )
    );

    /*
     * CUDA reports memoryClockRate in kHz.
     *
     * Approximate theoretical DDR bandwidth:
     *
     * 2 × memory clock × bus width / 8
     */
    double theoretical_memory_gb_s =
        2.0
        * static_cast<double>(
            prop.memoryClockRate
        )
        * 1000.0
        * (
            static_cast<double>(
                prop.memoryBusWidth
            )
            / 8.0
        )
        / 1.0e9;


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
        << "MAX_THREADS_PER_SM = "
        << prop.maxThreadsPerMultiProcessor
        << "\n";

    std::cout
        << "MAX_THREADS_PER_BLOCK = "
        << prop.maxThreadsPerBlock
        << "\n";

    std::cout
        << "MEMORY_BUS_WIDTH_BITS = "
        << prop.memoryBusWidth
        << "\n";

    std::cout
        << "MEMORY_CLOCK_KHZ = "
        << prop.memoryClockRate
        << "\n";

    std::cout
        << "L2_CACHE_MIB = "
        << std::fixed
        << std::setprecision(3)
        << static_cast<double>(
            prop.l2CacheSize
        )
        / (1024.0 * 1024.0)
        << "\n";

    std::cout
        << "GLOBAL_MEMORY_GIB = "
        << static_cast<double>(
            prop.totalGlobalMem
        )
        / (
            1024.0
            * 1024.0
            * 1024.0
        )
        << "\n";

    std::cout
        << "APPROX_THEORETICAL_MEMORY_GB_S = "
        << std::setprecision(3)
        << theoretical_memory_gb_s
        << "\n";


    return {
        prop.multiProcessorCount,
        prop.warpSize,
        prop.maxThreadsPerMultiProcessor,
        prop.memoryBusWidth,
        prop.memoryClockRate,
        theoretical_memory_gb_s
    };
}


template<typename LaunchFn>
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
        cudaEventCreate(
            &start
        )
    );

    CUDA_CHECK(
        cudaEventCreate(
            &stop
        )
    );


    CUDA_CHECK(
        cudaEventRecord(
            start
        )
    );

    for (int i = 0; i < repeats; ++i) {
        launch();
    }

    CUDA_CHECK(
        cudaEventRecord(
            stop
        )
    );

    CUDA_CHECK(
        cudaEventSynchronize(
            stop
        )
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
        cudaEventDestroy(
            start
        )
    );

    CUDA_CHECK(
        cudaEventDestroy(
            stop
        )
    );


    return
        static_cast<double>(
            total_ms
        )
        / repeats;
}


struct Result {
    std::string experiment;
    std::string variant;

    int parameter;

    double arithmetic_intensity;

    double time_ms;

    double effective_gib_s;

    double gflop_s;

    int registers_per_thread;

    int active_blocks_per_sm;

    double occupancy_percent;
};


template<int ITERS>
Result run_intensity_test(
    const float* input,
    float* output,
    size_t n,
    const DeviceInfo& device
) {
    int grid =
        static_cast<int>(
            (
                n
                + BLOCK_SIZE
                - 1
            )
            / BLOCK_SIZE
        );


    double ms =
        benchmark_kernel(
            [&]() {
                arithmetic_intensity_kernel<
                    ITERS
                ><<<
                    grid,
                    BLOCK_SIZE
                >>>(
                    input,
                    output,
                    n
                );
            },
            5,
            20
        );


    /*
     * 8 bytes useful memory traffic:
     * one FP32 read + one FP32 write.
     */
    double bytes =
        static_cast<double>(
            n
        )
        * 8.0;


    double gib =
        bytes
        / (
            1024.0
            * 1024.0
            * 1024.0
        );


    double seconds =
        ms / 1000.0;


    double effective_gib_s =
        gib / seconds;


    /*
     * Four FMAs per iteration.
     * One FMA = 2 FLOPs.
     *
     * = 8 FLOPs per iteration.
     */
    double flop =
        static_cast<double>(
            n
        )
        * static_cast<double>(
            ITERS
        )
        * 8.0;


    double gflop_s =
        flop
        / seconds
        / 1.0e9;


    double arithmetic_intensity =
        static_cast<double>(
            ITERS
        );


    cudaFuncAttributes attr{};

    CUDA_CHECK(
        cudaFuncGetAttributes(
            &attr,
            arithmetic_intensity_kernel<
                ITERS
            >
        )
    );


    int active_blocks = 0;

    CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            arithmetic_intensity_kernel<
                ITERS
            >,
            BLOCK_SIZE,
            0
        )
    );


    int active_threads =
        active_blocks
        * BLOCK_SIZE;


    double occupancy =
        100.0
        * static_cast<double>(
            active_threads
        )
        / static_cast<double>(
            device.max_threads_per_sm
        );


    if (occupancy > 100.0) {
        occupancy =
            100.0;
    }


    return {
        "arithmetic_intensity",
        "fma_sweep",
        ITERS,
        arithmetic_intensity,
        ms,
        effective_gib_s,
        gflop_s,
        attr.numRegs,
        active_blocks,
        occupancy
    };
}


float read_element(
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


    std::cout
        << LINE
        << "\n"
        << "GPU DEVICE CHARACTERIZATION"
        << "\n"
        << LINE
        << "\n";


    DeviceInfo device =
        print_device_info();


    /*
     * 2^26 float elements:
     *
     * 256 MiB input
     * 256 MiB output
     *
     * Large enough to be far beyond L2 cache.
     */
    constexpr size_t STREAM_N =
        1ULL << 26;


    constexpr size_t COMPUTE_N =
        1ULL << 24;


    float* input = nullptr;
    float* output = nullptr;


    size_t stream_bytes =
        STREAM_N
        * sizeof(float);


    CUDA_CHECK(
        cudaMalloc(
            &input,
            stream_bytes
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &output,
            stream_bytes
        )
    );


    int init_grid =
        static_cast<int>(
            (
                STREAM_N
                + BLOCK_SIZE
                - 1
            )
            / BLOCK_SIZE
        );


    init_kernel<<<
        init_grid,
        BLOCK_SIZE
    >>>(
        input,
        STREAM_N
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    std::vector<Result> results;


    /*
     * =====================================================================
     * EXPERIMENT 1 — STREAMING MEMORY BANDWIDTH
     * =====================================================================
     */

    std::cout
        << "\n"
        << LINE
        << "\n"
        << "EXPERIMENT 1 — STREAMING GLOBAL MEMORY BANDWIDTH"
        << "\n"
        << LINE
        << "\n";


    int stream_grid =
        static_cast<int>(
            (
                STREAM_N
                + BLOCK_SIZE
                - 1
            )
            / BLOCK_SIZE
        );


    double copy_ms =
        benchmark_kernel(
            [&]() {
                streaming_copy_kernel<<<
                    stream_grid,
                    BLOCK_SIZE
                >>>(
                    input,
                    output,
                    STREAM_N
                );
            },
            10,
            100
        );


    /*
     * Useful bytes:
     *
     * input read  = 4 bytes
     * output write = 4 bytes
     */
    double copy_bytes =
        static_cast<double>(
            STREAM_N
        )
        * 8.0;


    double copy_gib =
        copy_bytes
        / (
            1024.0
            * 1024.0
            * 1024.0
        );


    double copy_gib_s =
        copy_gib
        / (
            copy_ms
            / 1000.0
        );


    double copy_gb_s =
        copy_bytes
        / (
            copy_ms
            / 1000.0
        )
        / 1.0e9;


    double theoretical_efficiency =
        100.0
        * copy_gb_s
        / device.theoretical_memory_gb_s;


    std::cout
        << "STREAM_ELEMENTS = "
        << STREAM_N
        << "\n";

    std::cout
        << "INPUT_MIB = "
        << stream_bytes
           / (1024.0 * 1024.0)
        << "\n";

    std::cout
        << "OUTPUT_MIB = "
        << stream_bytes
           / (1024.0 * 1024.0)
        << "\n";

    std::cout
        << "COPY_KERNEL_MS = "
        << std::fixed
        << std::setprecision(6)
        << copy_ms
        << "\n";

    std::cout
        << "OBSERVED_MEMORY_GIB_S = "
        << std::setprecision(3)
        << copy_gib_s
        << "\n";

    std::cout
        << "OBSERVED_MEMORY_GB_S = "
        << copy_gb_s
        << "\n";

    std::cout
        << "THEORETICAL_MEMORY_GB_S = "
        << device.theoretical_memory_gb_s
        << "\n";

    std::cout
        << "APPROX_MEMORY_BANDWIDTH_EFFICIENCY = "
        << theoretical_efficiency
        << "%\n";


    results.push_back({
        "memory_bandwidth",
        "stream_copy",
        0,
        0.0,
        copy_ms,
        copy_gib_s,
        0.0,
        0,
        0,
        0.0
    });


    /*
     * =====================================================================
     * EXPERIMENT 2 — ARITHMETIC INTENSITY SWEEP
     * =====================================================================
     */

    std::cout
        << "\n"
        << LINE
        << "\n"
        << "EXPERIMENT 2 — ARITHMETIC INTENSITY SWEEP"
        << "\n"
        << LINE
        << "\n";


    std::cout
        << "COMPUTE_ELEMENTS = "
        << COMPUTE_N
        << "\n";

    std::cout
        << "BLOCK_SIZE = "
        << BLOCK_SIZE
        << "\n";

    std::cout
        << "\n";

    std::cout
        << "ITERS"
        << "\t"
        << "AI(F/B)"
        << "\t"
        << "time_ms"
        << "\t"
        << "GiB/s"
        << "\t"
        << "GFLOP/s"
        << "\t"
        << "regs/thread"
        << "\t"
        << "blocks/SM"
        << "\t"
        << "occupancy"
        << "\n";


    auto r1 =
        run_intensity_test<1>(
            input,
            output,
            COMPUTE_N,
            device
        );


    auto r4 =
        run_intensity_test<4>(
            input,
            output,
            COMPUTE_N,
            device
        );


    auto r16 =
        run_intensity_test<16>(
            input,
            output,
            COMPUTE_N,
            device
        );


    auto r64 =
        run_intensity_test<64>(
            input,
            output,
            COMPUTE_N,
            device
        );


    auto r256 =
        run_intensity_test<256>(
            input,
            output,
            COMPUTE_N,
            device
        );


    auto r1024 =
        run_intensity_test<1024>(
            input,
            output,
            COMPUTE_N,
            device
        );


    results.push_back(r1);
    results.push_back(r4);
    results.push_back(r16);
    results.push_back(r64);
    results.push_back(r256);
    results.push_back(r1024);


    for (
        const auto& r : results
    ) {
        if (
            r.experiment
            != "arithmetic_intensity"
        ) {
            continue;
        }


        std::cout
            << r.parameter
            << "\t"
            << std::fixed
            << std::setprecision(1)
            << r.arithmetic_intensity
            << "\t"
            << std::setprecision(6)
            << r.time_ms
            << "\t"
            << std::setprecision(2)
            << r.effective_gib_s
            << "\t"
            << std::setprecision(2)
            << r.gflop_s
            << "\t"
            << r.registers_per_thread
            << "\t\t"
            << r.active_blocks_per_sm
            << "\t\t"
            << std::setprecision(1)
            << r.occupancy_percent
            << "%"
            << "\n";
    }


    /*
     * =====================================================================
     * EXPERIMENT 3 — EMPIRICAL PERFORMANCE MODEL
     * =====================================================================
     */

    std::cout
        << "\n"
        << LINE
        << "\n"
        << "EXPERIMENT 3 — EMPIRICAL ROOFLINE-LIKE MODEL"
        << "\n"
        << LINE
        << "\n";


    double max_gflop_s = 0.0;


    for (
        const auto& r : results
    ) {
        if (
            r.experiment
            == "arithmetic_intensity"
        ) {
            max_gflop_s =
                std::max(
                    max_gflop_s,
                    r.gflop_s
                );
        }
    }


    /*
     * Convert observed streaming memory bandwidth
     * to bytes/s.
     */
    double memory_bytes_per_s =
        copy_gib_s
        * 1024.0
        * 1024.0
        * 1024.0;


    double compute_flop_per_s =
        max_gflop_s
        * 1.0e9;


    double empirical_ridge_point =
        compute_flop_per_s
        / memory_bytes_per_s;


    std::cout
        << "OBSERVED_STREAMING_MEMORY_GIB_S = "
        << std::fixed
        << std::setprecision(3)
        << copy_gib_s
        << "\n";


    std::cout
        << "MAX_OBSERVED_FP32_FMA_GFLOP_S = "
        << max_gflop_s
        << "\n";


    std::cout
        << "EMPIRICAL_RIDGE_POINT_FLOP_PER_BYTE = "
        << empirical_ridge_point
        << "\n";


    std::cout
        << "\n"
        << "INTERPRETATION:"
        << "\n";


    std::cout
        << "AI well below ridge point:"
        << "\n"
        << "    likely memory-side limited"
        << "\n";


    std::cout
        << "AI near ridge point:"
        << "\n"
        << "    transition region"
        << "\n";


    std::cout
        << "AI well above ridge point:"
        << "\n"
        << "    increasingly compute-side limited"
        << "\n";


    /*
     * =====================================================================
     * VALIDATION
     * =====================================================================
     */

    std::cout
        << "\n"
        << LINE
        << "\n"
        << "EXPERIMENT 4 — CORRECTNESS / SANITY"
        << "\n"
        << LINE
        << "\n";


    float sample =
        read_element(
            output,
            12345
        );


    bool finite =
        std::isfinite(
            sample
        );


    std::cout
        << "OUTPUT_SAMPLE = "
        << sample
        << "\n";


    std::cout
        << "OUTPUT_IS_FINITE = "
        << (
            finite
            ? "TRUE"
            : "FALSE"
        )
        << "\n";


    if (!finite) {
        std::cerr
            << "OUTPUT_SANITY = FAILED\n";

        return 2;
    }


    std::cout
        << "OUTPUT_SANITY = SUCCESS\n";


    /*
     * =====================================================================
     * CSV
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
        << "parameter,"
        << "arithmetic_intensity_flop_per_byte,"
        << "time_ms,"
        << "effective_gib_s,"
        << "gflop_s,"
        << "registers_per_thread,"
        << "active_blocks_per_sm,"
        << "occupancy_percent\n";


    for (
        const auto& r : results
    ) {
        csv
            << r.experiment
            << ","
            << r.variant
            << ","
            << r.parameter
            << ","
            << std::fixed
            << std::setprecision(3)
            << r.arithmetic_intensity
            << ","
            << std::setprecision(6)
            << r.time_ms
            << ","
            << std::setprecision(3)
            << r.effective_gib_s
            << ","
            << r.gflop_s
            << ","
            << r.registers_per_thread
            << ","
            << r.active_blocks_per_sm
            << ","
            << r.occupancy_percent
            << "\n";
    }


    csv.close();


    CUDA_CHECK(
        cudaFree(
            input
        )
    );


    CUDA_CHECK(
        cudaFree(
            output
        )
    );


    /*
     * =====================================================================
     * KNOWLEDGE CHECKPOINT
     * =====================================================================
     */

    std::cout
        << "\n"
        << LINE
        << "\n"
        << "P1-M4 KNOWLEDGE CHECKPOINT"
        << "\n"
        << LINE
        << "\n";


    std::cout << R"(
GPU PERFORMANCE MODEL

A GPU workload can often be understood by asking:

    How much computation is performed
    for every byte moved from memory?


ARITHMETIC INTENSITY

Arithmetic Intensity:

    FLOPs
    -----
    Bytes

Low arithmetic intensity:

    small amount of computation
    large amount of memory traffic

Typical tendency:

    memory-bound


High arithmetic intensity:

    large amount of computation
    relative to memory traffic

Typical tendency:

    compute-bound


EXAMPLE

Vector Copy:

    read 4 bytes
    write 4 bytes

but almost no arithmetic.

This is strongly memory-oriented.


FMA SWEEP

In this experiment every element performs:

    one global load
    one global store

while we progressively increase:

    1
    4
    16
    64
    256
    1024

groups of FP32 FMA operations.

Memory traffic stays approximately constant.

Compute work increases dramatically.

This lets us observe the transition from:

    memory-dominated
        ↓
    transition region
        ↓
    compute-dominated


ROOFLINE IDEA

A simplified performance ceiling is:

    attainable performance
        =
    min(
        compute ceiling,
        memory bandwidth × arithmetic intensity
    )

This is the core idea behind the Roofline model.


OCCUPANCY

Occupancy describes how many warps can be active
relative to the hardware maximum.

But:

    higher occupancy != automatically faster

Performance can also depend on:

    instruction-level parallelism
    registers
    memory latency
    memory bandwidth
    shared memory
    cache behavior
    instruction throughput


WHY THIS MATTERS FOR LLM SYSTEMS

Different LLM phases have different performance characteristics.

Large GEMMs:

    relatively high arithmetic intensity
    → often compute-oriented

Decode with KV Cache:

    relatively less computation per byte accessed
    → often much more memory-bandwidth-sensitive

Attention:

    can involve substantial data movement

FlashAttention:

    improves performance largely by reducing
    expensive HBM traffic through tiling and fusion

Quantization:

    reduces memory traffic and model footprint

Kernel fusion:

    reduces intermediate memory traffic
    and kernel-launch overhead


IMPORTANT LIMITATION

The FMA sweep in this experiment uses ordinary FP32 CUDA-core
operations to demonstrate arithmetic intensity.

It is NOT a Tensor Core peak-performance benchmark.

Later Transformer GEMM experiments will separately study:

    FP16 / BF16
    Tensor Cores
    GEMM
    Attention
    LLM inference
)" << std::endl;


    std::cout
        << LINE
        << "\n"
        << "FINAL VERDICT"
        << "\n"
        << LINE
        << "\n";


    std::cout
        << "STREAMING_MEMORY_BANDWIDTH_TEST = SUCCESS\n"
        << "ARITHMETIC_INTENSITY_SWEEP = SUCCESS\n"
        << "OCCUPANCY_CHARACTERIZATION = SUCCESS\n"
        << "EMPIRICAL_PERFORMANCE_MODEL = SUCCESS\n"
        << "OUTPUT_SANITY = SUCCESS\n"
        << "STRUCTURED_CSV_RESULT = SUCCESS\n"
        << "VERDICT = READY_FOR_P1_M5_TRANSFORMER_LLM_INFERENCE\n";


    return 0;
}
