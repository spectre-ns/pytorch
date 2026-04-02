#pragma once
#include <ATen/native/DispatchStub.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/Scalar.h>
#include <c10/util/TypeSafeSignMath.h>

namespace at{
    struct TensorIterator;
    struct TensorIteratorBase;
}

namespace at::native {
    using test_operation_signature = void(*)(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out1, Tensor& out2);

    DECLARE_DISPATCH(test_operation_signature, test_operation_3_stub)
}
