#include <cassert>

#include <iomanip>
#include <iostream>
#include <memory>

#include "layers/Dense.hpp"
#include "layers/ReLu.hpp"
#include "layers/Sigmoid.hpp"

#include "loss/MSEloss.hpp"

#include "math/Matrix.hpp"

#include "nn/Sequential.hpp"

namespace {

void print_matrix(const char *name, const Matrix &m) {
  std::cout << name << " (" << m.rows() << "x" << m.col() << "):\n";
  const float *ptr = m.data();
  for (std::size_t r = 0; r < m.rows(); ++r) {
    std::cout << "  [ ";
    for (std::size_t c = 0; c < m.col(); ++c) {
      std::cout << std::fixed << std::setprecision(4) << ptr[r * m.col() + c]
                << " ";
    }
    std::cout << "]\n";
  }
  std::cout << "\n";
}

} // namespace

int main() {
  // 1. Construct XOR input batch: shape (4 x 2)
  Matrix X(4, 2);
  float *x_data = X.data();

  x_data[0 * 2 + 0] = 0.0f;
  x_data[0 * 2 + 1] = 0.0f;
  x_data[1 * 2 + 0] = 0.0f;
  x_data[1 * 2 + 1] = 1.0f;
  x_data[2 * 2 + 0] = 1.0f;
  x_data[2 * 2 + 1] = 0.0f;
  x_data[3 * 2 + 0] = 1.0f;
  x_data[3 * 2 + 1] = 1.0f;

  // 2. Construct XOR target batch: shape (4 x 1)
  Matrix Y(4, 1);
  float *y_data = Y.data();

  y_data[0] = 0.0f;
  y_data[1] = 1.0f;
  y_data[2] = 1.0f;
  y_data[3] = 0.0f;

  print_matrix("Input X", X);
  print_matrix("Target Y", Y);

  // 3. Define Neural Network Architecture
  Sequential model;
  model.add(std::make_unique<Dense>(2, 4));
  model.add(std::make_unique<Sigmoid>());
  model.add(std::make_unique<Dense>(4, 1));
  model.add(std::make_unique<Sigmoid>());

  // Initial predictions before training
  Matrix initial_pred = model.forward(X);
  print_matrix("Predictions BEFORE Training", initial_pred);

  // 4. Training Loop setup
  MSEloss loss_fn;
  const float learning_rate = 1.0f;
  const int epochs = 100000;

  std::cout << "Starting Training (" << epochs
            << " epochs, learning rate = " << learning_rate << ")...\n";
  std::cout << "--------------------------------------------------\n";

  for (int epoch = 1; epoch <= epochs; ++epoch) {
    // Forward pass
    Matrix y_pred = model.forward(X);

    // Compute loss
    float loss = loss_fn.forward(y_pred, Y);

    // Backward pass
    Matrix loss_grad = loss_fn.backward(y_pred, Y);
    model.backward(loss_grad);

    // Update parameters
    model.update_parameters(learning_rate);

    if (epoch == 1 || epoch % 1000 == 0 || epoch == epochs) {
      std::cout << "Epoch " << std::setw(5) << epoch
                << " | MSE Loss: " << std::fixed << std::setprecision(6) << loss
                << "\n";
    }
  }

  std::cout << "--------------------------------------------------\n\n";

  // 5. Final Evaluation
  Matrix final_pred = model.forward(X);
  print_matrix("Predictions AFTER Training", final_pred);

  // Verify accuracy threshold for XOR
  const float *preds = final_pred.data();
  bool success = true;
  for (std::size_t i = 0; i < 4; ++i) {
    float expected = y_data[i];
    float actual = preds[i];
    float rounded = actual >= 0.5f ? 1.0f : 0.0f;
    if (rounded != expected) {
      success = false;
    }
  }

  if (success) {
    std::cout << "XOR Neural Network training completed successfully!\n";
  } else {
    std::cout << "XOR Neural Network training did not converge completely.\n";
  }

  return 0;
}
