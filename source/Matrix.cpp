#include <cstddef>
#include <vector>

#include "Matrix.hpp"

Matrix::Matrix(std::size_t rows, std::size_t col, float init_value)
    : rows_(rows), col_(col), data_(rows * col, init_value) {}

Matrix::Matrix(std::size_t rows, std::size_t col, std::vector<float> data)
    : rows_(rows), col_(col), data_(std::move(data)) {}

float &Matrix::operator()(std::size_t r, std::size_t c) {
  return data_[r * col_ + c];
}

const float &Matrix::operator()(std::size_t r, std::size_t c) const {
  return data_[r * col_ + c];
}

Matrix Matrix::transpose() const {
  std::vector<float> transposed(rows_ * col_);

  for (std::size_t r = 0; r < rows_; ++r) {
    for (std::size_t c = 0; c < col_; ++c) {
      transposed[c * rows_ + r] = data_[r * col_ + c];
    }
  }

  return Matrix(col_, rows_, std::move(transposed));
}

Matrix &Matrix::operator+=(const Matrix &rhs) {
  for (std::size_t i = 0; i < data_.size(); ++i) {
    data_[i] += rhs.data_[i];
  }
  return *this;
}

Matrix &Matrix::operator-=(const Matrix &rhs) {
  for (std::size_t i = 0; i < data_.size(); ++i) {
    data_[i] -= rhs.data_[i];
  }
  return *this;
}

Matrix Matrix::hadamard(const Matrix &a, const Matrix &b) {
  Matrix result(a.rows_, a.col_);
  for (std::size_t i = 0; i < a.data_.size(); ++i) {
    result.data_[i] = a.data_[i] * b.data_[i];
  }
  return result;
}
