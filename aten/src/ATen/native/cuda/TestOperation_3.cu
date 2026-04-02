#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Math.cuh>
#include <ATen/NumericUtils.h>
#include <ATen/native/TestOperation.h>

namespace at::native {
    void test_operation_3_kernel_cuda(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out1, Tensor& out2){
        //TODO: Impl
    }

    REGISTER_DISPATCH(test_operation_3_stub, &test_operation_3_kernel_cuda)
}