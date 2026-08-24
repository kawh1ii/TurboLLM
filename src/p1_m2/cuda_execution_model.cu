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


__global__ void mapping_kernel(
    int* block_index,
    int* thread_index,
    int* global_index,
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        block_index[gid] = blockIdx.x;
        thread_index[gid] = threadIdx.x;
        global_index[gid] = gid;
    }
}


__global__ void initialize_kernel(
    float* a,
    float* b,
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        a[gid] = static_cast<float>(gid % 100) * 0.01f;
        b[gid] = static_cast<float>((gid + 7) % 100) * 0.02f;
    }
}


__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* c,
    int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < n) {
        c[gid] = a[gid] + b[gid];
    }
}


void print_device_info() {
    int device = 0;

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "GPU_NAME = "
              << prop.name
              << "\n";

    std::cout << "COMPUTE_CAPABILITY = "
              << prop.major
              << "."
              << prop.minor
              << "\n";

    std::cout << "SM_COUNT = "
              << prop.multiProcessorCount
              << "\n";

    std::cout << "WARP_SIZE = "
              << prop.warpSize
              << "\n";

    std::cout << "MAX_THREADS_PER_BLOCK = "
              << prop.maxThreadsPerBlock
              << "\n";

    std::cout << "MAX_THREADS_DIM_X = "
              << prop.maxThreadsDim[0]
              << "\n";

    std::cout << "MAX_GRID_DIM_X = "
              << prop.maxGridSize[0]
              << "\n";

    std::cout << "GLOBAL_MEMORY_GIB = "
              << std::fixed
              << std::setprecision(3)
              << static_cast<double>(prop.totalGlobalMem)
                 / (1024.0 * 1024.0 * 1024.0)
              << "\n";
}


void execution_mapping_demo() {
    constexpr int N = 32;
    constexpr int BLOCK = 8;

    const int GRID = (N + BLOCK - 1) / BLOCK;

    int* d_block = nullptr;
    int* d_thread = nullptr;
    int* d_global = nullptr;

    CUDA_CHECK(cudaMalloc(&d_block, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_thread, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_global, N * sizeof(int)));

    mapping_kernel<<<GRID, BLOCK>>>(
        d_block,
        d_thread,
        d_global,
        N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_block(N);
    std::vector<int> h_thread(N);
    std::vector<int> h_global(N);

    CUDA_CHECK(cudaMemcpy(
        h_block.data(),
        d_block,
        N * sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        h_thread.data(),
        d_thread,
        N * sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        h_global.data(),
        d_global,
        N * sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    std::cout
        << "GRID_SIZE = "
        << GRID
        << "\n";

    std::cout
        << "BLOCK_SIZE = "
        << BLOCK
        << "\n";

    std::cout << "\n";

    std::cout
        << "global_thread"
        << "\t"
        << "blockIdx.x"
        << "\t"
        << "threadIdx.x"
        << "\n";

    for (int i = 0; i < N; ++i) {
        std::cout
            << h_global[i]
            << "\t\t"
            << h_block[i]
            << "\t\t"
            << h_thread[i]
            << "\n";
    }

    CUDA_CHECK(cudaFree(d_block));
    CUDA_CHECK(cudaFree(d_thread));
    CUDA_CHECK(cudaFree(d_global));
}


struct BenchmarkResult {
    int block_size;
    int grid_size;
    float kernel_ms;
    double bandwidth_gib_s;
};


BenchmarkResult benchmark_vector_add(
    const float* a,
    const float* b,
    float* c,
    int n,
    int block_size
) {
    const int grid_size =
        (n + block_size - 1) / block_size;

    constexpr int WARMUP = 20;
    constexpr int REPEATS = 200;

    for (int i = 0; i < WARMUP; ++i) {
        vector_add_kernel<<<grid_size, block_size>>>(
            a,
            b,
            c,
            n
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < REPEATS; ++i) {
        vector_add_kernel<<<grid_size, block_size>>>(
            a,
            b,
            c,
            n
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &total_ms,
        start,
        stop
    ));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const float kernel_ms =
        total_ms / REPEATS;

    // Vector add:
    //
    // read A = 4 bytes
    // read B = 4 bytes
    // write C = 4 bytes
    //
    // approximately 12 bytes / element.
    const double bytes =
        static_cast<double>(n) * 3.0 * sizeof(float);

    const double gib =
        bytes / (1024.0 * 1024.0 * 1024.0);

    const double seconds =
        static_cast<double>(kernel_ms) / 1000.0;

    const double bandwidth =
        gib / seconds;

    return {
        block_size,
        grid_size,
        kernel_ms,
        bandwidth
    };
}


void validate_result(
    const float* a,
    const float* b,
    const float* c
) {
    constexpr int CHECK_N = 16;

    std::vector<float> ha(CHECK_N);
    std::vector<float> hb(CHECK_N);
    std::vector<float> hc(CHECK_N);

    CUDA_CHECK(cudaMemcpy(
        ha.data(),
        a,
        CHECK_N * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        hb.data(),
        b,
        CHECK_N * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaMemcpy(
        hc.data(),
        c,
        CHECK_N * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    for (int i = 0; i < CHECK_N; ++i) {
        const float expected =
            ha[i] + hb[i];

        if (std::fabs(hc[i] - expected) > 1e-5f) {
            std::cerr
                << "VALIDATION FAILED AT "
                << i
                << "\n";

            std::exit(2);
        }
    }

    std::cout
        << "VECTOR_ADD_VALIDATION = SUCCESS"
        << "\n";
}


int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr
            << "Usage: "
            << argv[0]
            << " result.csv"
            << std::endl;

        return 1;
    }

    const std::string csv_path = argv[1];

    const std::string LINE(120, '=');

    std::cout
        << LINE
        << "\n"
        << "CUDA DEVICE CHARACTERIZATION"
        << "\n"
        << LINE
        << "\n";

    print_device_info();

    std::cout
        << "\n"
        << LINE
        << "\n"
        << "EXPERIMENT 1 — GRID / BLOCK / THREAD MAPPING"
        << "\n"
        << LINE
        << "\n";

    execution_mapping_demo();

    std::cout
        << "\n"
        << LINE
        << "\n"
        << "EXPERIMENT 2 — VECTOR ADD BLOCK SIZE"
        << "\n"
        << LINE
        << "\n";

    constexpr int N = 1 << 24;

    const size_t bytes =
        static_cast<size_t>(N) * sizeof(float);

    std::cout
        << "VECTOR_ELEMENTS = "
        << N
        << "\n";

    std::cout
        << "ARRAY_SIZE_MIB_EACH = "
        << std::fixed
        << std::setprecision(3)
        << static_cast<double>(bytes)
           / (1024.0 * 1024.0)
        << "\n";

    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;

    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMalloc(&c, bytes));

    constexpr int INIT_BLOCK = 256;

    const int init_grid =
        (N + INIT_BLOCK - 1)
        / INIT_BLOCK;

    initialize_kernel<<<init_grid, INIT_BLOCK>>>(
        a,
        b,
        N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> block_sizes = {
        64,
        128,
        256,
        512,
        1024
    };

    std::vector<BenchmarkResult> results;

    std::cout
        << "\n"
        << "block_size"
        << "\t"
        << "warps/block"
        << "\t"
        << "grid_size"
        << "\t"
        << "kernel_ms"
        << "\t"
        << "effective_GiB/s"
        << "\n";

    for (int block_size : block_sizes) {
        BenchmarkResult result =
            benchmark_vector_add(
                a,
                b,
                c,
                N,
                block_size
            );

        results.push_back(result);

        std::cout
            << result.block_size
            << "\t\t"
            << result.block_size / 32
            << "\t\t"
            << result.grid_size
            << "\t\t"
            << std::fixed
            << std::setprecision(6)
            << result.kernel_ms
            << "\t"
            << std::setprecision(3)
            << result.bandwidth_gib_s
            << "\n";
    }

    validate_result(a, b, c);

    std::ofstream csv(csv_path);

    if (!csv.is_open()) {
        std::cerr
            << "FAILED TO OPEN CSV: "
            << csv_path
            << "\n";

        return 3;
    }

    csv
        << "block_size,"
        << "warps_per_block,"
        << "grid_size,"
        << "kernel_ms,"
        << "effective_gib_s\n";

    for (const auto& result : results) {
        csv
            << result.block_size
            << ","
            << result.block_size / 32
            << ","
            << result.grid_size
            << ","
            << std::fixed
            << std::setprecision(6)
            << result.kernel_ms
            << ","
            << std::setprecision(3)
            << result.bandwidth_gib_s
            << "\n";
    }

    csv.close();

    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));
    CUDA_CHECK(cudaFree(c));

    std::cout
        << "\n"
        << LINE
        << "\n"
        << "P1-M2 KNOWLEDGE CHECKPOINT"
        << "\n"
        << LINE
        << "\n";

    std::cout << R"(
CUDA kernel execution hierarchy:

Grid
 └── Block
      └── Warp
           └── Thread

Global thread index in a 1D kernel:

gid = blockIdx.x * blockDim.x + threadIdx.x

Important observations:

1. A kernel launch creates a Grid.

2. A Grid contains many Blocks.

3. A Block contains many Threads.

4. NVIDIA hardware schedules threads in groups called Warps.

5. On this GPU:
   warp size = 32 threads.

6. Block size determines how many threads are grouped into each block.

7. Different block sizes can change performance.

8. More threads per block is NOT automatically faster.

9. Vector Add performs very little arithmetic but moves a lot of memory.

10. Therefore Vector Add is primarily a memory-bandwidth-oriented workload.

This is the first concrete example of:

    compute-bound
    vs
    memory-bound

which will become critical later for LLM inference.
)" << std::endl;

    std::cout
        << LINE
        << "\n"
        << "FINAL VERDICT"
        << "\n"
        << LINE
        << "\n";

    std::cout
        << "CUDA_COMPILE = SUCCESS\n"
        << "CUDA_KERNEL_EXECUTION = SUCCESS\n"
        << "GRID_BLOCK_THREAD_MAPPING = SUCCESS\n"
        << "VECTOR_ADD_VALIDATION = SUCCESS\n"
        << "BLOCK_SIZE_BENCHMARK = SUCCESS\n"
        << "STRUCTURED_CSV_RESULT = SUCCESS\n"
        << "VERDICT = READY_FOR_P1_M3_CUDA_MEMORY_AND_WARP_BEHAVIOR\n";

    return 0;
}
