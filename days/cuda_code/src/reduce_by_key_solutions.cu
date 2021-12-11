#include <cassert>

#include <thrust/reduce.h>

#include "solution_utils.h"

__global__ void reduce_by_key_atomic_kernel(const float *inp,
                                            const int64_t *keys, float *dest,
                                            int size) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= size) {
    return;
  }
  atomicAdd(&dest[keys[i]], inp[i]);
}

void reduce_by_key_atomic(const float *gpu_inp, const int64_t *gpu_keys,
                          float *gpu_dest, int size) {
  int block_size = 512;
  reduce_by_key_atomic_kernel<<<ceil_divide(size, block_size), block_size>>>(
      gpu_inp, gpu_keys, gpu_dest, size);
  CUDA_SYNC_CHK();
}

template <typename F>
std::vector<float> reduce_by_keys_vec(const std::vector<float> &inp,
                                      const std::vector<int64_t> &keys,
                                      const F &f) {
  assert(inp.size() == keys.size());
  assert(!inp.empty());
  assert(std::is_sorted(keys.begin(), keys.end()));
  float *gpu_inp = copy_to_gpu(inp.data(), inp.size());
  int64_t *gpu_keys = copy_to_gpu(keys.data(), keys.size());

  int max_key = keys[keys.size() - 1];
  int out_size = max_key + 1;

  float *dest;
  CUDA_ERROR_CHK(cudaMalloc(&dest, out_size * sizeof(float)));
  CUDA_ERROR_CHK(cudaMemset(dest, 0, out_size * sizeof(float)));

  f(gpu_inp, gpu_keys, dest, inp.size());

  auto out = copy_from_gpu(dest, out_size);

  CUDA_ERROR_CHK(cudaFree(gpu_inp));
  CUDA_ERROR_CHK(cudaFree(gpu_keys));
  CUDA_ERROR_CHK(cudaFree(dest));

  return out;
}

std::vector<int64_t> random_keys(int64_t size) {
  std::random_device rd;
  std::default_random_engine engine(rd());
  std::geometric_distribution<int64_t> dist(0.8);
  int64_t running = 0;
  std::vector<int64_t> out(size);
  std::generate(out.begin(), out.end(), [&] {
    running += dist(engine);
    return running;
  });

  return out;
}

std::vector<float> reduce_by_keys_cpu(const std::vector<float> &inp,
                                      const std::vector<int64_t> &keys) {
  assert(!keys.empty());
  assert(inp.size() == keys.size());

  int max_key = keys[keys.size() - 1];
  int out_size = max_key + 1;

  std::vector<float> out(out_size);

  for (size_t i = 0; i < inp.size(); ++i) {
    out[keys[i]] += inp[i];
  }

  return out;
}

bool vecs_near(const std::vector<float> &l, const std::vector<float> &r) {
  if (l.size() != r.size()) {
    return false;
  }
  for (size_t i = 0; i < l.size(); ++i) {
    if (std::abs(l[i] - r[i]) > 1e-4) {
      std::cout << "exit on diff: " << std::abs(l[i] - r[i]) << " at idx: " << i
                << std::endl;
      return false;
    }
  }
  return true;
}

template <typename F> void check_reduce_by_keys(const F &f) {
  std::vector<float> host_mem_single_inp{1.7f};
  std::vector<int64_t> host_mem_single_keys{0};
  std::cout << "single sum: "
            << reduce_by_keys_vec(host_mem_single_inp, host_mem_single_keys, f)
            << "\n";
  std::vector<int64_t> host_mem_single_other_keys{7};
  std::cout << "single sum at other index: "
            << reduce_by_keys_vec(host_mem_single_inp,
                                  host_mem_single_other_keys, f)
            << "\n";

  std::vector<float> few_filter_inp{1.8f, 12.f, -2.1f, 4.f, 3.f, 9.f, 10.f};
  std::vector<int64_t> few_filter_keys{0, 3, 3, 3, 3, 4, 8};
  std::cout << "few reduce by key: "
            << reduce_by_keys_vec(few_filter_inp, few_filter_keys, f) << "\n";

  for (int size : {513, 100000}) {
    auto inp = random_floats(-8.0f, 8.0f, size);
    auto keys = random_keys(size);
    auto cpu_vec = reduce_by_keys_cpu(inp, keys);
    auto gpu_vec = reduce_by_keys_vec(inp, keys, f);

    std::cout << "is near (size: " << size << "): " << std::boolalpha
              << vecs_near(cpu_vec, gpu_vec) << std::endl;
  }
}

int main() { check_reduce_by_keys(reduce_by_key_atomic); }
