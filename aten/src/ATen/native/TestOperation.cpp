#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <tuple>
#include <ATen/native/TestOperation.h>
#include <ATen/ops/test_operation_3.h>
#include <ATen/ops/test_operation_3_native.h>
#include <ATen/TensorMeta.h>

#include <ATen/Tensor.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/tanh.h>
#include <ATen/ops/add.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/sub.h>

namespace at::meta
{
    //Macro expansion failure until next codegen run!
    TORCH_META_FUNC2(test_operation_3, Tensor) (const Tensor& q, const Tensor& k, const Tensor& v) {
        //Perform all the meta logic.  
    }
}

namespace at::native
{
    std::tuple<Tensor, Tensor> TestOperation(const Tensor& q, const Tensor& k, const Tensor& v) 
    {
        auto x = at::matmul(q, k.transpose(0, 1));
        auto a = at::tanh(x);
        auto o = at::matmul(a, v);
        return std::make_tuple(o, a);
    }

    std::tuple<Tensor, Tensor> TestOperation_2(const Tensor& q, const Tensor& k, const Tensor& v) 
    {
        return at::native::TestOperation(q,k,v);
    }

    DEFINE_DISPATCH(test_operation_3_stub);
    TORCH_IMPL_FUNC(test_operation_3_out) (const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& out1, const Tensor& out2) 
    {
        //launch the stub based on the device
        //test_operation_3_stub(device_type(), q, k, v);
    }
}