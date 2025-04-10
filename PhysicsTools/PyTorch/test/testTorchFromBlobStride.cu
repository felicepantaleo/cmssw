#include <cstdlib>
#include <cuda/std/array>
#include <cuda/std/ranges>
#include <iostream>
#include <chrono>
#include <array>

#include <cuda_runtime.h>

#include <torch/torch.h>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

template <std::size_t N>
torch::Tensor array_to_tensor(torch::Device device, int* arr, const long int* size) {
  long int arr_size[N];
  long int arr_stride[N];
  std::copy(size, size+N, arr_size);
  std::copy(size, size+N, arr_stride);

  std::shift_right(std::begin(arr_stride), std::end(arr_stride), 1);
  arr_stride[0] = 1;
  arr_stride[N-1] *= arr_stride[N-2];

  auto options = torch::TensorOptions().dtype(torch::kInt).device(device).pinned_memory(true);
  torch::Tensor tensor = torch::from_blob(arr, arr_size, arr_stride, options);

  return tensor;
}

template <std::size_t N>
void print_column_major(int* arr, const long int* size) {
  if (N == 2) {
    for (int i = 0; i < size[0]; i++) {
      for (int j = 0; j < size[1]; j++) {
          std::cout << arr[i + j*size[0]] << " ";
      }
      std::cout << std::endl;
    } 
  } else if (N == 3) {
    for (int i = 0; i < size[0]; i++) {
      std::cout << "(" << i << ", .., ..)" << std::endl;
      for (int j = 0; j < size[1]; j++) {
        for (int k = 0; k < size[2]; k++) {
          std::cout << arr[i + j*size[0] + k*size[0]*size[1]] << " ";
        }
        std::cout << std::endl;
      } 
      std::cout << std::endl;
    }
  }
}


int main(int argc, char* argv[]) {  
  torch::Device device(torch::kCUDA);

  int a_cpu[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  // 2 Dimensional Example
  long int a_shape[] = {4, 6};

  // 3 Dimensional Example
  // const long int a_shape[] = {4, 3, 2};

  const size_t dims = sizeof(a_shape) / sizeof(long int);

  // Prints array in correct form.
  print_column_major<dims>(a_cpu, a_shape);

  int *a_gpu;
  cudaMalloc(&a_gpu, sizeof(a_cpu));
  cudaMemcpy(a_gpu, a_cpu, sizeof(a_cpu), cudaMemcpyHostToDevice);

  // bad behaviour
  auto options = torch::TensorOptions().dtype(torch::kInt).device(device).pinned_memory(true);
  std::cout << "Converting vector to Torch tensors on CPU without stride" << std::endl;
  torch::Tensor tensor = torch::from_blob(a_gpu, a_shape, options);
  std::cout << tensor << std::endl;  

  // Correct Transposition to get to smae dimensions as column major.
  std::cout << "Correct Tensor with Transpose" << std::endl;
  long int a_size[dims];
  std::copy(a_shape, a_shape+dims, a_size);
  std::reverse(std::begin(a_size), std::end(a_size));
  tensor = torch::from_blob(a_gpu, a_size, options);

  tensor = torch::transpose(tensor, 0, dims-1);
  std::cout << tensor << std::endl;

  // Use stride to read correctly.
  std::cout << "Converting vector to Torch tensors on CPU with stride" << std::endl;
  std::cout << array_to_tensor<dims>(device, a_gpu, a_shape) << std::endl;

  long int b_shape[] = {500, 1000};
  int b[b_shape[0]][b_shape[1]];

  for (int i = 0; i < b_shape[0]; i++) {
    for (int j = 0; j < b_shape[1]; j++) {
      b[i][j] = rand();
    }
  } 

  int *b_gpu;
  cudaMalloc(&b_gpu, b_shape[0] * b_shape[1] * sizeof(int));
  cudaMemcpy(b_gpu, b, b_shape[0] * b_shape[1] * sizeof(int), cudaMemcpyHostToDevice);

  std::cout << "Benchmark stride and transpose:" << std::endl;

  auto t1 = high_resolution_clock::now();
  const size_t dim_b = sizeof(b_shape) / sizeof(long int);
  torch::Tensor tensor_stride = array_to_tensor<dim_b>(device, b_gpu, b_shape);
  auto t2 = high_resolution_clock::now();
  duration<double, std::milli> ms_double = t2 - t1;
  std::cout << "Stride:" << ms_double.count() << "ms\n";

  t1 = high_resolution_clock::now();
  const size_t dim_b2 = sizeof(b_shape) / sizeof(long int);
  long int b_size[dim_b2];
  std::copy(b_shape, b_shape+dim_b2, b_size);
  std::reverse(std::begin(a_size), std::end(a_size));

  std::reverse(std::begin(b_shape), std::end(b_shape));
  torch::Tensor tensor_transp = torch::from_blob(b_gpu, b_size, options);
  tensor_transp = torch::transpose(tensor_transp, 0, dim_b2-1);
  t2 = high_resolution_clock::now();
  ms_double = t2 - t1;
  std::cout << "Transpose:" << ms_double.count() << "ms\n";

  return 0;
}
