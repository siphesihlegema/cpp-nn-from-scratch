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
