#include <cstddef>

#include "Layer.hpp"

#include "math/Matrix.hpp"

class Dense : public Layer {
private:
  Matrix input_cache_;
  Matrix W_;
  Matrix B_;
  Matrix dW_;
  Matrix dB_;

public:
  Dense(std::size_t input_feat, std::size_t output_feat);
  Matrix forward(const Matrix &input) override;
  Matrix backward(const Matrix &output_grad) override;
  void update_parameters(float learning_rate) override;
};
