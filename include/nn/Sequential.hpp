#include <memory>

#include "layers/Layer.hpp"

class Sequential {
private:
  std::vector<std::unique_ptr<Layer>> layers_;

public:
  void add(std::unique_ptr<Layer> layer);
  Matrix forward(Matrix x);
  Matrix backward(Matrix x);
  void update_parameters(float learning_rate);
};
