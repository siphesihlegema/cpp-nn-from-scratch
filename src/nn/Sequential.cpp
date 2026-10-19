#include <utility>

#include "nn/Sequential.hpp"

void Sequential::add(std::unique_ptr<Layer> layer) {
  layers_.push_back(std::move(layer));
}

Matrix Sequential::forward(Matrix x) {
  for (auto &layer : layers_) {
    x = layer->forward(x);
  }
  return x;
}
