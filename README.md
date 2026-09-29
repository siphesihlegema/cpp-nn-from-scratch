# C++ Neural Network Engine from Scratch

## What Is This Project?
This project is an ongoing implementation of a deep learning engine written in C++20 from absolute first principles, without using third-party linear algebra or machine learning libraries. It covers: low-level contiguous memory management and cache-friendly matrix operations up to automatic differentiation, modular network layers, loss functions, and model training.

## Technical Architecture & Scope
* **Custom Linear Algebra (`Matrix`):** Contiguous 1D buffer allocation (`std::vector<float>`) using row-major mapping `r * col + c`, featuring cache-aware matrix multiplication ($i \to k \to j$ loops), bias broadcasting, and gradient reduction.
* **Modular Layer System (`Layer`):** An extensible interface for chaining operations—including affine transformations (`Dense`) and non-linear activations (`ReLU`, `Sigmoid`)—with forward-state caching for exact backward gradient computation.
* **Optimization & Training Pipeline:** Custom loss functions (MSE, numerically stabilized cross-entropy) and sequential model orchestration designed to verify convergence on non-linear benchmarks.

## Building & Testing

```bash
cmake --build build && ./build/main.out
