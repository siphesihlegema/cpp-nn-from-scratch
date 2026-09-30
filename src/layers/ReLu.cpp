#include "layers/ReLu.hpp"

#include "math/Matrix.hpp"

Matrix ReLu::forward(const Matrix &input) {
  input_cache_ = input;
  Matrix out(input.rows(), input.col());

  const float *in_ptr = input.data();
  float *out_ptr = out.data();

  for (std::size_t i = 0; i < input.size(); ++i) {
    if (in_ptr[i] > 0.0f) {
      out_ptr[i] = in_ptr[i];
    }
  }
  return out;
}

Matrix ReLu::backward(const Matrix &output_grad) {
  Matrix dX(output_grad.rows(), output_grad.col());

  const float *grad_ptr = output_grad.data();
  const float *cache_ptr = input_cache_.data();
  float *dx_ptr = dX.data();

  for (std::size_t i = 0; i < output_grad.size(); ++i) {
    if (cache_ptr[i] > 0.0f) {
      dx_ptr[i] = grad_ptr[i];
    }
  }

  return dX;
}
