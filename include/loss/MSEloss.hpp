#include "loss/Loss.hpp"

#include "math/Matrix.hpp"

class MSEloss : public Loss {
public:
  float forward(const Matrix &pred, const Matrix &target) override;
  Matrix backward(const Matrix &pred, const Matrix &target) override;
};
