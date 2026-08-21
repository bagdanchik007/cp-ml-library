#include "ml/ml.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace ml;

namespace {

void test_matrix_squared_norm() {
    std::cout << "[TEST] Matrix squared norm ... ";

    const Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    assert(matrix.squared_norm() == 30.0);

    std::cout << "OK\n";
}

void test_matrix_norm() {
    std::cout << "[TEST] Matrix norm ... ";

    const Matrix matrix = {
        {3.0, 4.0}
    };

    assert(std::abs(matrix.norm() - 5.0) < 1e-9);

    std::cout << "OK\n";
}

void test_matrix_norm_zero_matrix() {
    std::cout << "[TEST] Zero matrix norm ... ";

    const Matrix matrix(2, 2);

    assert(matrix.squared_norm() == 0.0);
    assert(matrix.norm() == 0.0);

    std::cout << "OK\n";
}

void test_matrix_squared_norm_negative_values() {
    std::cout << "[TEST] Squared norm with negative values ... ";

    const Matrix matrix = {
        {-1.0, -2.0},
        {-3.0, -4.0}
    };

    assert(matrix.squared_norm() == 30.0);

    std::cout << "OK\n";
}

} // namespace

int main() {
    std::cout << "Running matrix norm tests...\n\n";

    test_matrix_squared_norm();
    test_matrix_norm();
    test_matrix_norm_zero_matrix();
    test_matrix_squared_norm_negative_values();

    std::cout << "\nAll matrix norm tests passed.\n";

    return 0;
}