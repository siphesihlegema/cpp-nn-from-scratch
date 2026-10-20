#include "layers/Dense.hpp"

#include "cstddef"

#include "math/Matrix.hpp"

Dense::Dense(std::size_t input_feat, std::size_t output_feat)
    : W_(Matrix::xavier(input_feat, output_feat)), B_(1, output_feat),
      dW_(input_feat, output_feat), dB_(1, output_feat) {}

Matrix Dense::forward(const Matrix &input) {
  input_cache_ = input;
  Matrix Z = Matrix::multiply(input, W_);
  Z.add_bias(B_);
  return Z;
}

Matrix Dense::backward(const Matrix &output_grad) {
  dW_ = Matrix::multiply(input_cache_.transpose(), output_grad);
  dB_ = output_grad.sum_rows();
  return Matrix::multiply(output_grad, W_.transpose());
}
