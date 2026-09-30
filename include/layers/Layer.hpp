#include "Matrix.hpp"

class Layer {
public:
  virtual ~Layer() = default;
  virtual Matrix forward(const Matrix &input);
  virtual Matrix backward(const Matrix &output_grad);
  virtual void update_parameters(float learning_rate);
};
