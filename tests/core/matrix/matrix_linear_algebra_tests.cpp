#include "ml/ml.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace ml;

namespace {

void test_matrix_trace() {
    std::cout << "[TEST] Matrix trace ... ";

    const Matrix matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    assert(matrix.trace() == 15.0);

    std::cout << "OK\n";
}

void test_matrix_determinant_1x1() {
    std::cout << "[TEST] Matrix determinant 1x1 ... ";

    const Matrix matrix = {
        {5.0}
    };

    assert(matrix.determinant() == 5.0);

    std::cout << "OK\n";
}

void test_matrix_determinant_2x2() {
    std::cout << "[TEST] Matrix determinant 2x2 ... ";

    const Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    assert(matrix.determinant() == -2.0);

    std::cout << "OK\n";
}

void test_matrix_determinant_3x3() {
    std::cout << "[TEST] Matrix determinant 3x3 ... ";

    const Matrix matrix = {
        {6.0, 1.0, 1.0},
        {4.0, -2.0, 5.0},
        {2.0, 8.0, 7.0}
    };

    assert(matrix.determinant() == -306.0);

    std::cout << "OK\n";
}

void test_matrix_inverse_1x1() {
    std::cout << "[TEST] Matrix inverse 1x1 ... ";

    const Matrix matrix = {
        {4.0}
    };

    const Matrix result = matrix.inverse();

    assert(std::abs(result(0, 0) - 0.25) < 1e-12);

    std::cout << "OK\n";
}

void test_matrix_inverse_2x2() {
    std::cout << "[TEST] Matrix inverse 2x2 ... ";

    const Matrix matrix = {
        {4.0, 7.0},
        {2.0, 6.0}
    };

    const Matrix result = matrix.inverse();

    assert(std::abs(result(0, 0) - 0.6) < 1e-12);
    assert(std::abs(result(0, 1) + 0.7) < 1e-12);
    assert(std::abs(result(1, 0) + 0.2) < 1e-12);
    assert(std::abs(result(1, 1) - 0.4) < 1e-12);

    std::cout << "OK\n";
}

void test_matrix_inverse_singular() {
    std::cout << "[TEST] Matrix inverse rejects singular matrix ... ";

    const Matrix matrix = {
        {1.0, 2.0},
        {2.0, 4.0}
    };

    bool exception_thrown = false;

    try {
        [[maybe_unused]] const Matrix result = matrix.inverse();
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);

    std::cout << "OK\n";
}

} // namespace

int main() {
    std::cout << "Running matrix linear algebra tests...\n\n";

    test_matrix_trace();
    test_matrix_determinant_1x1();
    test_matrix_determinant_2x2();
    test_matrix_determinant_3x3();
    test_matrix_inverse_1x1();
    test_matrix_inverse_2x2();
    test_matrix_inverse_singular();
    std::cout << "\nAll matrix linear algebra tests passed.\n";

    return 0;
}