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
