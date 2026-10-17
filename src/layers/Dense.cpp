#include "layers/Dense.hpp"

#include "math/Matrix.hpp"

Matrix Dense::forward(const Matrix &input) {
  input_cache_ = input;
  Matrix Z = Matrix::multiply(input, W_);
  Z.add_bias(B_);
  return Z;
}
