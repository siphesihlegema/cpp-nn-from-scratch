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
