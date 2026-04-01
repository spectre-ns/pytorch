#include <tuple>

#include <ATen/Tensor.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/tanh.h>
#include <ATen/ops/add.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/sub.h>


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
}