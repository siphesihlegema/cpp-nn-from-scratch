#include "Layer.hpp"

#include "math/Matrix.hpp"

class ReLu : public Layer {
private:
  Matrix input_cache_;

public:
  Matrix forward(const Matrix &input) override;
  Matrix backward(const Matrix &output_grad) override;
};
