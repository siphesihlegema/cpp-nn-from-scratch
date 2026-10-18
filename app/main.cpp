#include <cassert>
#include <cmath>

#include <iostream>

#include "math/Matrix.hpp"

void print_matrix(const std::string &name, const Matrix &m) {
  std::cout << name << " (" << m.rows() << "x" << m.col() << "):\n";
  for (std::size_t r = 0; r < m.rows(); ++r) {
    std::cout << "  [ ";
    for (std::size_t c = 0; c < m.col(); ++c) {
      std::cout << m(r, c) << " ";
    }
    std::cout << "]\n";
  }
  std::cout << "\n";
}

int main() {
  std::cout << "=== Running Matrix Tests ===\n\n";

  // 1. Constructors & Data Access
  Matrix m_default;
  assert(m_default.rows() == 0 && m_default.col() == 0);

  Matrix m_init(2, 3, 4.0f);
  assert(m_init.rows() == 2 && m_init.col() == 3);
  assert(m_init(0, 0) == 4.0f && m_init(1, 2) == 4.0f);

  m_init(1, 1) = 9.0f;
  assert(m_init(1, 1) == 9.0f);
  print_matrix("m_init (modified at [1,1])", m_init);

  // 2. Vector Constructor & Raw Pointer Access
  std::vector<float> raw_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Matrix a(2, 3, raw_data);
  assert(a.data()[1] == 2.0f);
  print_matrix("Matrix A", a);

  // 3. Transpose
  Matrix a_t = a.transpose();
  assert(a_t.rows() == 3 && a_t.col() == 2);
  assert(a_t(0, 1) == 4.0f && a_t(2, 0) == 3.0f && a_t(2, 1) == 6.0f);
  print_matrix("Matrix A Transposed", a_t);

  // 4. Matrix Multiplication: A (2x3) * A_T (3x2) -> Result (2x2)
  Matrix mat_mul = Matrix::multiply(a, a_t);
  assert(mat_mul.rows() == 2 && mat_mul.col() == 2);
  assert(mat_mul(0, 0) == 14.0f);
  assert(mat_mul(0, 1) == 32.0f);
  assert(mat_mul(1, 0) == 32.0f);
  assert(mat_mul(1, 1) == 77.0f);
  print_matrix("A * A_T (Multiply)", mat_mul);

  // 5. In-place & Binary Addition / Subtraction
  Matrix b(2, 3, {6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f});
  Matrix c = a + b;
  for (std::size_t i = 0; i < c.rows() * c.col(); ++i) {
    assert(c.data()[i] == 7.0f);
  }

  Matrix d = b - a;
  assert(d(0, 0) == 5.0f && d(1, 2) == -5.0f);

  a += b;
  assert(a(0, 0) == 7.0f);
  a -= b;
  assert(a(0, 0) == 1.0f);

  // 6. Scalar Multiplication
  Matrix scaled = a * 2.0f;
  assert(scaled(0, 0) == 2.0f && scaled(1, 2) == 12.0f);

  Matrix scaled_pre = 3.0f * a;
  assert(scaled_pre(0, 0) == 3.0f && scaled_pre(1, 2) == 18.0f);

  scaled *= 0.5f;
  assert(scaled(0, 0) == 1.0f);

  // 7. Hadamard Product (Element-wise)
  Matrix hada = Matrix::hadamard(a, b);
  assert(hada(0, 0) == 6.0f);
  assert(hada(0, 1) == 10.0f);
  assert(hada(0, 2) == 12.0f);
  assert(hada(1, 0) == 12.0f);
  assert(hada(1, 1) == 10.0f);
  assert(hada(1, 2) == 6.0f);
  print_matrix("Hadamard Product (A * B)", hada);

  // 8. Random Uniform Initialization
  float low = -2.0f, high = 2.0f;
  Matrix rand_mat = Matrix::random_uniform(5, 5, low, high);
  assert(rand_mat.rows() == 5 && rand_mat.col() == 5);
  for (std::size_t i = 0; i < 25; ++i) {
    assert(rand_mat.data()[i] >= low && rand_mat.data()[i] <= high);
  }
  print_matrix("Random Uniform (5x5, [-2, 2])", rand_mat);

  // 9. Xavier Initialization
  std::size_t fan_in = 100, fan_out = 50;
  float expected_limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
  Matrix xavier_mat = Matrix::xavier(fan_out, fan_in);

  assert(xavier_mat.rows() == fan_out && xavier_mat.col() == fan_in);
  for (std::size_t i = 0; i < fan_out * fan_in; ++i) {
    assert(xavier_mat.data()[i] >= -expected_limit &&
           xavier_mat.data()[i] <= expected_limit);
  }
  std::cout
      << "Xavier Matrix (50x100) bound check passed: all elements within [-"
      << expected_limit << ", +" << expected_limit << "]\n\n";

  // 10. Bias Addition Broadcast (Forward Pass)
  // Input: (3, 2) matrix, Bias: (1, 2) row vector
  Matrix z_mat(3, 2, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Matrix bias_vec(1, 2, {0.5f, -0.1f});

  print_matrix("Pre-activations before bias", z_mat);
  print_matrix("Bias row vector", bias_vec);

  z_mat.add_bias(bias_vec);
  print_matrix("Pre-activations after add_bias", z_mat);

  assert(std::abs(z_mat(0, 0) - 1.5f) < 1e-5f);
  assert(std::abs(z_mat(0, 1) - 1.9f) < 1e-5f);
  assert(std::abs(z_mat(1, 0) - 3.5f) < 1e-5f);
  assert(std::abs(z_mat(1, 1) - 3.9f) < 1e-5f);
  assert(std::abs(z_mat(2, 0) - 5.5f) < 1e-5f);
  assert(std::abs(z_mat(2, 1) - 5.9f) < 1e-5f);
  std::cout << "Bias addition broadcast assertions passed.\n\n";

  // 11. Row-wise Summation (Backward Pass)
  // Collapse (3, 2) incoming gradients down to (1, 2) bias gradient
  Matrix grad_z(3, 2, {0.1f, -0.2f, 0.4f, 0.5f, -0.3f, 0.1f});

  print_matrix("Incoming grad_z", grad_z);
  Matrix grad_b = grad_z.sum_rows();
  print_matrix("Collapsed grad_b (sum_rows)", grad_b);

  assert(grad_b.rows() == 1 && grad_b.col() == 2);
  assert(std::abs(grad_b(0, 0) - 0.2f) < 1e-5f); // 0.1 + 0.4 - 0.3 = 0.2
  assert(std::abs(grad_b(0, 1) - 0.4f) < 1e-5f); // -0.2 + 0.5 + 0.1 = 0.4
  std::cout << "Row-wise summation assertions passed.\n\n";

  std::cout << "All assertions passed successfully.\n";
  return 0;
}
