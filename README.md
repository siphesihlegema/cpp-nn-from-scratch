# C++20 Neural Network Engine from Scratch

![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)
![Build](https://img.shields.io/badge/CMake-3.20%2B-green.svg)
![Dependencies](https://img.shields.io/badge/Dependencies-None-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A lightweight, zero-dependency deep learning framework written in modern C++20. 

## Features
**Custom Linear Algebra Core (`Matrix`):** Contiguous 1D row-major memory buffer (`std::vector<float>`) with cache-friendly `i -> k -> j` loop ordering for General Matrix Multiplication (GEMM), row-wise bias broadcasting, reductions, and Xavier/Glorot weight initialization.

**PyTorch-Inspired Modular API (`Sequential`):** Stack layers polymorphically using `std::unique_ptr<Layer>` with clean `forward()`, `backward()`, and `update_parameters()` passes.

**Hand-Derived Analytical Gradients:** Every layer and loss function computes exact analytical Jacobians during the backward pass—no black-box autograd overhead.

**Zero External Dependencies:** Compiles cleanly with any standard C++20 compiler and CMake.

## Architecture & Mathematical Foundations

### Memory Layout & Tensor Shapes
All matrices are stored in a flat, contiguous 1D array using row-major indexing ($r \cdot \text{cols} + c$) to maximize CPU L1/L2 cache line utilization. Mini-batches are processed with one sample per row:
**Input Batch ($X$):** shape $(B \times D_{\text{in}})$
**Weights ($W$):** shape $(D_{\text{in}} \times D_{\text{out}})$, initialized via Xavier/Glorot uniform bounds $\pm\sqrt{\frac{6}{D_{\text{in}} + D_{\text{out}}}}$
**Biases ($b$):** shape $(1 \times D_{\text{out}})$, broadcast across all $B$ rows

### The `Layer` Contract
Every neural network component implements the base `Layer` interface, caching intermediate state during `forward()` required to compute upstream and parameter gradients during `backward()`:

```cpp
class Layer {
public:
  virtual ~Layer() = default;
  virtual Matrix forward(const Matrix &input) = 0;        // Compute activations & cache state
  virtual Matrix backward(const Matrix &output_grad) = 0; // Compute dL/dX and parameter grads
  virtual void update_parameters(float learning_rate) {}  // Apply gradient descent step
};
```

## Example (Training XOR)

```cpp
#include "layers/Dense.hpp"
#include "layers/ReLu.hpp"
#include "layers/Sigmoid.hpp"
#include "loss/MSEloss.hpp"
#include "math/Matrix.hpp"
#include "nn/Sequential.hpp"

int main() {
  // 1. Define architecture
  Sequential model;
  model.add(std::make_unique<Dense>(2, 4));
  model.add(std::make_unique<Sigmoid>());
  model.add(std::make_unique<Dense>(4, 1));
  model.add(std::make_unique<Sigmoid>());

  MSEloss loss_fn;
  const float lr = 1.0f;

  // 2. Training loop (X is 4x2 input batch, Y is 4x1 target batch)
  for (int epoch = 1; epoch <= 10000; ++epoch) {
    Matrix y_pred = model.forward(X);
    float loss = loss_fn.forward(y_pred, Y);

    Matrix grad = loss_fn.backward(y_pred, Y);
    model.backward(grad);
    model.update_parameters(lr);
  }
}
```

---

## Structure

```text
cpp-nn-from-scratch/
├── app/
│   └── main.cpp                 # Training entry point & XOR benchmark
├── include/
│   ├── layers/
│   │   ├── Layer.hpp            # Abstract base layer interface
│   │   ├── Dense.hpp            # Fully connected linear layer
│   │   ├── ReLu.hpp             # Rectified Linear Unit activation
│   │   └── Sigmoid.hpp          # Logistic activation
│   ├── loss/
│   │   ├── Loss.hpp             # Abstract base loss interface
│   │   └── MSEloss.hpp          # Mean Squared Error loss
│   ├── math/
│   │   └── Matrix.hpp           # Contiguous row-major 2D matrix engine
│   └── nn/
│       └── Sequential.hpp       # Model container for chaining layers
├── src/
│   ├── layers/                  # Layer implementations
│   ├── loss/                    # Loss implementations
│   ├── math/                    # Matrix arithmetic & GEMM kernels
│   └── nn/                      # Sequential container implementation
├── CMakeLists.txt               # CMake build configuration
└── LICENSE                      # MIT License
```

---

## Build and Run

**Prerequisites:** A C++20 compatible compiler (`GCC 10+` or `Clang 12+`) and `CMake 3.20+`.

```bash
# 1. Configure in Release mode for compiler optimizations (-O3)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# 2. Compile the engine and demo executable
cmake --build build

# 3. Run the training benchmark
./build/main.out
```

---

## Roadmap & Upcoming Optimizations

The core training engine is complete and verified. Current development is focused on low-level hardware optimization and expanding deep learning capabilities:

### Low-Level Performance & Memory (Phase 5)
- [ ] **SIMD Vectorization:** Implement AVX2 FMA intrinsics (`_mm256_fmadd_ps`) for 8-wide float vectorized GEMM and element-wise operations.
- [ ] **Multi-Threading:** Parallelize outer GEMM loops and batch processing with OpenMP and cache-blocked loop tiling.
- [ ] **Zero-Allocation Training Loop:** Preallocate persistent scratchpad buffers and introduce in-place destination operations (`Matrix::multiply(A, B, out)`) to eliminate heap allocations during training.
- [ ] **GFLOP/s Micro-Benchmarking:** Benchmark suite comparing naive `i -> j -> k`, cache-aware `i -> k -> j`, and SIMD/OpenMP kernels across matrix sizes from $64 \times 64$ to $1024 \times 1024$.

### Deep Learning Features
- [ ] **Fused `CrossEntropyLoss`:** Numerically stable Softmax ($\max(z)$ shifted) fused with negative log-likelihood for multi-class classification ($\frac{\partial L}{\partial z} = \hat{Y} - Y$).
- [ ] **Stateful Optimizers:** Decouple parameter updates from layers to support `SGD` with Momentum and `Adam`.
- [ ] **Data Pipeline & Serialization:** Add mini-batch row slicing/shuffling (`Matrix::slice_rows`) and binary model weight saving/loading.

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
