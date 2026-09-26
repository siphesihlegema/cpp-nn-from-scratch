#pragma once

#include <cstddef>
#include <vector>

class Matrix {
private:
  std::size_t rows_{};
  std::size_t col_{};
  std::vector<float> data_;

public:
  Matrix() = default;
  Matrix(std::size_t rows, std::size_t col, float init_value = 0.0f);
  Matrix(std::size_t rows, std::size_t col, std::vector<float> data);

  // getters
  std::size_t rows() const { return rows_; }
  std::size_t col() const { return col_; }
  float *data() { return data_.data(); }
  const float *data() const { return data_.data(); }

  // data access
  float &operator()(std::size_t r, std::size_t c);
  const float &operator()(std::size_t r, std::size_t c) const;

  // matrix transpose
  Matrix transpose() const;

  // in place operations
  Matrix &operator+=(const Matrix &rhs);
  Matrix &operator-=(const Matrix &rhs);
  Matrix &operator*=(float scalar);

  // Hadamard product
  static Matrix hadamard(const Matrix &a, const Matrix &b);

  // cache-aware multiplication
  static Matrix multiply(const Matrix &a, const Matrix &b);
};

// binary operations
inline Matrix operator+(Matrix lhs, const Matrix &rhs) {
  lhs += rhs;
  return lhs;
}

inline Matrix operator-(Matrix lhs, const Matrix &rhs) {
  lhs -= rhs;
  return lhs;
}

inline Matrix operator*(Matrix mat, float scalar) {
  mat *= scalar;
  return mat;
}

inline Matrix operator*(float scalar, Matrix &mat) {
  mat *= scalar;
  return mat;
}
