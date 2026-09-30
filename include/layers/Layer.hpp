#include "math/Matrix.hpp"

class Layer {
public:
  virtual ~Layer() = default;
  virtual Matrix forward(const Matrix &input) = 0;
  virtual Matrix backward(const Matrix &output_grad) = 0;
  virtual void update_parameters(float learning_rate) { (void)learning_rate; }
};
