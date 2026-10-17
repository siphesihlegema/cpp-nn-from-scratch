#pragma once

#include "layers/Layer.hpp"
#include "math/Matrix.hpp"

class Sigmoid : public Layer {
private:
  Matrix output_cache_;

public:
  Matrix forward(const Matrix &input) override;
  Matrix backward(const Matrix &output_grad) override;
};
