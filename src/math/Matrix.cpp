#include <cmath>

#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

#include "math/Matrix.hpp"

std::string shape_str(std::size_t r, std::size_t c) {
  return "(" + std::to_string(r) + "x" + std::to_string(c) + ")";
}

Matrix::Matrix(std::size_t rows, std::size_t col, float init_value)
    : rows_(rows), col_(col), data_(rows * col, init_value) {}

Matrix::Matrix(std::size_t rows, std::size_t col, std::vector<float> data)
    : rows_(rows), col_(col), data_(std::move(data)) {
  if (data_.size() != rows_ * col_) {
    throw std::invalid_argument(
        "Matrix constructor size mismatch: specified shape " +
        shape_str(rows_, col_) + " requires " + std::to_string(rows_ * col_) +
        " elements, but passed vector has " + std::to_string(data_.size()) +
        " elements.");
  }
}

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
  if (rows_ != rhs.rows_ || col_ != rhs.col_) {
    throw std::invalid_argument(
        "Matrix += shape mismatch: " + shape_str(rows_, col_) + " vs " +
        shape_str(rhs.rows_, rhs.col_));
  }

  const float *ptr = rhs.data();
  float *out_ptr = data_.data();

  for (std::size_t i = 0; i < this->size(); ++i) {
    out_ptr[i] += ptr[i];
  }
  return *this;
}

Matrix &Matrix::operator-=(const Matrix &rhs) {
  if (rows_ != rhs.rows_ || col_ != rhs.col_) {
    throw std::invalid_argument(
        "Matrix -= shape mismatch: " + shape_str(rows_, col_) + " vs " +
        shape_str(rhs.rows_, rhs.col_));
  }
  const float *ptr = rhs.data();
  float *out_ptr = data_.data();

  for (std::size_t i = 0; i < this->size(); ++i) {
    out_ptr[i] -= ptr[i];
  }
  return *this;
}

Matrix &Matrix::operator*=(float scalar) {
  float *ptr = data_.data();

  for (std::size_t i = 0; i < this->size(); ++i) {
    ptr[i] *= scalar;
  }
  return *this;
}

Matrix Matrix::hadamard(const Matrix &a, const Matrix &b) {
  if (a.rows_ != b.rows_ || a.col_ != b.col_) {
    throw std::invalid_argument(
        "Matrix hadamard shape mismatch: " + shape_str(a.rows_, a.col_) +
        " vs " + shape_str(b.rows_, b.col_));
  }
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
  if (a.col() != b.rows()) {
    throw std::invalid_argument(
        "Matrix multiply inner dimension mismatch: " +
        shape_str(a.rows(), a.col()) + " cannot multiply " +
        shape_str(b.rows(), b.col()) + ". (a.col must equal b.rows)");
  }
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
  if (bias.rows_ != 1 || bias.col_ != col_) {
    throw std::invalid_argument(
        "Bias broadcast dimension mismatch: target matrix shape is " +
        shape_str(rows_, col_) + ", but bias shape is " +
        shape_str(bias.rows_, bias.col_) + ". Expected (1, " +
        std::to_string(col_) + ")");
  }
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
