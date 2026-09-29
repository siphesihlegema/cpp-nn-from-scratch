#include <cmath>

#include <cstddef>
#include <random>
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

Matrix &Matrix::operator*=(float scalar) {
  float *ptr = data_.data();
  const std::size_t size = data_.size();
  for (std::size_t i = 0; i < size; ++i) {
    ptr[i] *= scalar;
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

Matrix Matrix::random_uniform(size_t r, size_t c, float low, float high) {
  std::vector<float> randomvec(r * c);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(low, high);

  std::size_t size(r * c);

  for (std::size_t i = 0; i < size; ++i) {
    randomvec[i] = dist(gen);
  }

  return Matrix(r, c, randomvec);
}

Matrix Matrix::xavier(size_t r, size_t c) {
  float limit = std::sqrt(6.0f / static_cast<float>(r + c));
  return Matrix::random_uniform(r, c, -limit, limit);
}

void Matrix::add_bias(const Matrix &bias) {
  const float *bias_ptr = bias.data();
  float *mat_ptr = data_.data();

  for (std::size_t r = 0; r < rows_; ++r) {
    std::size_t row_offset = r * col_;
    for (std::size_t c = 0; c < col_; ++c) {
      mat_ptr[row_offset + c] += bias_ptr[c];
    }
  }
}

Matrix Matrix::sum_rows() const {
  Matrix result(1, col_, 0.0f);

  const float *in_ptr = data_.data();
  float *out_ptr = result.data();

  for (std::size_t r = 0; r < rows_; ++r) {
    std::size_t row_offset = r * col_;
    for (std::size_t c = 0; c < col_; ++c) {
      out_ptr[c] += in_ptr[row_offset + c];
    }
  }

  return result;
}
