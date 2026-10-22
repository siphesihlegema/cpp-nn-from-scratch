#include <stdexcept>
#include "loss/MSEloss.hpp"

#include "math/Matrix.hpp"

float MSEloss::forward(const Matrix &pred, const Matrix &target) {
  const float *p = pred.data();
  const float *t = target.data();
  const std::size_t n = pred.size();

  float sum = 0.0f;

  for (std::size_t i = 0; i < n; ++i) {
    const float diff = p[i] - t[i];
    sum += diff * diff;
  }
  return sum / static_cast<float>(n);
}

Matrix MSEloss::backward(const Matrix &pred, const Matrix &target) {
  if (pred.rows() != target.rows() || pred.col() != target.col()) {
    throw std::invalid_argument("MSEloss shape mismatch in backward pass.");
  }

  Matrix grad(pred.rows(), pred.col());
  const float *p = pred.data();
  const float *t = target.data();
  float *g = grad.data();

  const std::size_t n = pred.size();
  const float factor = 2.0f / static_cast<float>(n);

  for (std::size_t i = 0; i < n; ++i) {
    g[i] = factor * (p[i] - t[i]);
  }

  return grad;
}

