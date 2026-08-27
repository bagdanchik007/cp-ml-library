#include "ml/core/matrix/matrix.hpp"

#include <chrono>
#include <iostream>

int main() {
    const ml::Matrix left = ml::Matrix::random(300, 300, -1.0, 1.0, 1);
    const ml::Matrix right = ml::Matrix::random(300, 300, -1.0, 1.0, 2);
    const auto start = std::chrono::steady_clock::now();
    const ml::Matrix result = left * right;
    const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cout << "matrix_multiply_300x300_ms=" << elapsed << " checksum=" << result(0, 0) << '\n';
}
