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

  const float *a_ptr = a.data();
  const float *b_ptr = b.data();
  float *out_ptr = result.data();

  const std::size_t size = a.rows_ * a.col_;

  for (std::size_t i = 0; i < size; ++i) {
    out_ptr[i] = a_ptr[i] * b_ptr[i];
  }
  return result;
}

Matrix Matrix::multiply(const Matrix &a, const Matrix &b) {

  std::size_t M = a.rows();
  std::size_t K = a.col();
  std::size_t N = b.col();

  Matrix result(M, N);

  const float *a_ptr = a.data();
  const float *b_ptr = b.data();
  float *c_ptr = result.data();

  for (std::size_t i = 0; i < M; ++i) {
    for (std::size_t k = 0; k < K; ++k) {
      const float a_ik = a_ptr[i * K + k];

      const std::size_t b_offset = k * N;
      const std::size_t c_offset = i * N;

      for (std::size_t j = 0; j < N; ++j) {
        c_ptr[c_offset + j] += a_ik * b_ptr[b_offset + j];
      }
    }
  }

  return result;
}
