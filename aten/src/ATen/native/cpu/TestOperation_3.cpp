#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/TestOperation.h>

#include <cmath>
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/Math.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/macros/Macros.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/generic_math.h>
#include <ATen/native/TestOperation.h>
#include <ATen/core/TensorBase.h>

namespace at::native {
    namespace{
    void test_operation_3_kernel(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out1, Tensor& out2) {
        // auto x = at::native::matmul(q, k.transpose(0, 1));
        // out2 = at::native::tanh_quantized_cpu(x);
        // out1 = at::native::matmul(out2, v);
    }
    }
    REGISTER_DISPATCH(test_operation_3_stub, &test_operation_3_kernel)
}