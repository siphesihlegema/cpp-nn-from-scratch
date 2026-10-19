#include <cassert>

#include <iomanip>
#include <iostream>
#include <memory>

#include "layers/Dense.hpp"
#include "layers/ReLu.hpp"
#include "layers/Sigmoid.hpp"

#include "math/Matrix.hpp"

#include "nn/Sequential.hpp"

namespace {

void print_matrix(const char *name, const Matrix &m) {
  std::cout << name << " (" << m.rows() << "x" << m.col() << "):\n";
  const float *ptr = m.data();
  for (std::size_t r = 0; r < m.rows(); ++r) {
    std::cout << "  [ ";
    for (std::size_t c = 0; c < m.col(); ++c) {
      std::cout << std::fixed << std::setprecision(4) << ptr[r * m.col() + c]
                << " ";
    }
    std::cout << "]\n";
  }
  std::cout << "\n";
}

} // namespace

int main() {
  // 1. Construct XOR input batch: shape (4 x 2)
  Matrix X(4, 2);
  float *x_data = X.data();

  x_data[0 * 2 + 0] = 0.0f;
  x_data[0 * 2 + 1] = 0.0f;
  x_data[1 * 2 + 0] = 0.0f;
  x_data[1 * 2 + 1] = 1.0f;
  x_data[2 * 2 + 0] = 1.0f;
  x_data[2 * 2 + 1] = 0.0f;
  x_data[3 * 2 + 0] = 1.0f;
  x_data[3 * 2 + 1] = 1.0f;

  print_matrix("Input X", X);

  Sequential model;
  model.add(std::make_unique<Dense>(2, 4));
  model.add(std::make_unique<ReLu>());
  model.add(std::make_unique<Dense>(4, 1));
  model.add(std::make_unique<Sigmoid>());

  Matrix y_pred = model.forward(X);
  assert(y_pred.rows() == 4 && y_pred.col() == 1 &&
         "y_pred shape mismatch: expected (4x1)");

  print_matrix("Output Prediction Y_pred", y_pred);

  return 0;
}
