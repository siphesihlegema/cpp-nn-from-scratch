#pragma once

#include <cstddef>
#include <stddef.h>
#include <vector>

class Matrix {
private:
  std::size_t rows_{};
  std::size_t col_{};
  std::vector<float> data_;

public:
  Matrix(std::size_t rows, std::size_t col, float init_value = 0.0f);
  Matrix(std::size_t rows, std::size_t col, std::vector<float> data);

  // getters
  std::size_t rows() { return rows_; }
  std::size_t col() { return col_; }
  std::vector<float> data() { return data_; }

  // data access
  float &operator()(std::size_t r, std::size_t c);
  const float &operator()(std::size_t r, std::size_t c) const;

  // matrix transpose
  Matrix transpose() const;

  // in place operations
  Matrix &operator+=(const Matrix &rhs);
  Matrix &operator-=(const Matrix &rhs);
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
