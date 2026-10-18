#pragma once

#include "math/Matrix.hpp"

class Loss {
public:
  virtual ~Loss() = default;
  virtual float forward(const Matrix &pred, const Matrix &target) = 0;
  virtual Matrix backward(const Matrix &pred, const Matrix &target) = 0;
};
