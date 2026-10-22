#include <ranges>
#include <utility>

#include "nn/Sequential.hpp"

void Sequential::add(std::unique_ptr<Layer> layer) {
  layers_.push_back(std::move(layer));
}

Matrix Sequential::forward(Matrix x) {
  for (auto &layer : layers_) {
    x = layer->forward(std::move(x));
  }
  return x;
}

Matrix Sequential::backward(Matrix grad) {
  for (auto &layer : layers_ | std::views::reverse) {
    grad = layer->backward(std::move(grad));
  }
  return grad;
}

void Sequential::update_parameters(float learning_rate) {
  for (auto &layer : layers_ | std::views::reverse) {
    layer->update_parameters(learning_rate);
  }
}
