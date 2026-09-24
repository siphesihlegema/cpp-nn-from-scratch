#include <iostream>

#include "../include/Matrix.hpp"

int main() {
  Matrix test(2, 2, {1, 2, 3, 4});

  Matrix trantest = test.transpose();

  Matrix test2(3, 3);

  std::cout << test(0, 1) << trantest(0, 1) << "\n";
  test += test2;

  std::cout << test(0, 1) << "\n";

  Matrix added = test + test2;
  added = added - test + trantest;
  std::cout << added(0, 0) << "\n";

  Matrix result = Matrix::hadamard(added, test2);
}
