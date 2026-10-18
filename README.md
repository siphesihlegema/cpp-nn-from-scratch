# C++ Neural Network Engine from Scratch

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![Build](https://img.shields.io/badge/CMake-3.20%2B-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A small neural network library written in C++20 with no external dependencies.

---

## How it works

A neural network is a stack of layers. Training repeats three steps:

1. **Forward pass:** data flows through the layers to produce a prediction.
2. **Backward pass (backpropagation):** the error flows back through the layers in reverse, and each layer works out how much it contributed.
3. **Update:** each layer adjusts its weights slightly to reduce the error.

Every layer in this project follows the same simple contract:

```cpp
class Layer {
public:
  virtual ~Layer() = default;
  virtual Matrix forward(const Matrix &input) = 0;        // compute the output
  virtual Matrix backward(const Matrix &output_grad) = 0; // compute the gradients
  virtual void update_parameters(float learning_rate) {}  // adjust the weights
};
```

---

## What's inside

**`Matrix`:** a simple 2D matrix class, stored in one flat array. It has the operations a network needs: multiply, add, transpose, adding a bias to every row, and random weight initialization.

**Layers:**

| Layer | What it does |
| :--- | :--- |
| `Dense` | Fully connected layer: `output = input × weights + bias` |
| `ReLu` | Activation that turns negative values into 0 |
| `Sigmoid` | Activation that squashes values into the range 0 to 1 |

The gradients for each layer are written by hand. This is not an automatic differentiation engine (yet).

---

## Project structure

```
cpp-nn-from-scratch/
├── app/
│   └── main.cpp        # Demo program
├── include/
│   ├── layers/         # Layer headers: Layer, Dense, ReLu, Sigmoid
│   └── math/
│       └── Matrix.hpp  # Matrix class
├── src/
│   ├── layers/         # Layer implementations
│   └── math/
│       └── Matrix.cpp  # Matrix implementation
├── CMakeLists.txt      # Build configuration
├── LICENSE             # MIT License```

---

## Build and run

C++20 compiler (GCC or Clang) and CMake 3.20 or newer.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release   # configure
cmake --build build                         # build
./build/main.out                            # run the demo
```

---

## License

MIT. See [LICENSE](LICENSE).
