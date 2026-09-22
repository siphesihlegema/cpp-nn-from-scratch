#include <cstddef>
#include <vector>

#include "../include/Matrix.hpp"

Matrix::Matrix(std::size_t rows, std::size_t col, float init_value)
    : rows_(rows), col_(col), data_(std::vector(rows * col, init_value)) {}

Matrix::Matrix(std::size_t rows, std::size_t col, std::vector<float> data)
    : rows_(rows), col_(col), data_(data) {}

float &Matrix::operator()(std::size_t r, std::size_t c) {
  return data_[r * col_ + c];
}

const float &Matrix::operator()(std::size_t r, std::size_t c) const {
  return data_[r * col_ + c];
}
