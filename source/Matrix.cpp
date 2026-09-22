#include <cstddef>
#include <vector>

#include "../include/Matrix.hpp"

Matrix::Matrix(std::size_t rows, std::size_t col, float init_value)
    : rows_(rows), col_(col),
      data_(std::move(std::vector(rows * col, init_value))) {}

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
