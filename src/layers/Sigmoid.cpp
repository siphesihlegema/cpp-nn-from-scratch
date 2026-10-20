#include <cmath>

#include "layers/Sigmoid.hpp"

#include "math/Matrix.hpp"

Matrix Sigmoid::forward(const Matrix &input) {
  Matrix out(input.rows(), input.col(), 0.0f);

  const float *in_ptr = input.data();
  float *out_ptr = out.data();

  for (std::size_t i = 0; i < input.size(); ++i) {
    out_ptr[i] = 1.0f / (1.0f + std::exp(-in_ptr[i]));
  }

  output_cache_ = out;
  return out;
}

Matrix Sigmoid::backward(const Matrix &output_grad) {
  Matrix dX(output_grad.rows(), output_grad.col());

  const float *grad_ptr = output_grad.data();
  const float *cache_ptr = output_cache_.data();
  float *dx_ptr = dX.data();

  for (std::size_t i = 0; i < output_grad.size(); ++i) {
    dx_ptr[i] = grad_ptr[i] * cache_ptr[i] * (1 - cache_ptr[i]);
  }
  return dX;
}
