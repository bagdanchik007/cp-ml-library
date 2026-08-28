#include "ml/ml.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace ml;

namespace {

void test_matrix_scalar_multiplication() {
    std::cout << "[TEST] Matrix scalar multiplication ... ";

    const Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const Matrix result = matrix * 2.0;

    assert(result(0, 0) == 2.0);
    assert(result(0, 1) == 4.0);
    assert(result(1, 0) == 6.0);
    assert(result(1, 1) == 8.0);

    std::cout << "OK\n";
}

void test_matrix_scalar_multiplication_negative() {
    std::cout << "[TEST] Matrix negative scalar multiplication ... ";

    const Matrix matrix = {
        {1.0, -2.0},
        {3.0, -4.0}
    };

    const Matrix result = matrix * -2.0;

    assert(result(0, 0) == -2.0);
    assert(result(0, 1) == 4.0);
    assert(result(1, 0) == -6.0);
    assert(result(1, 1) == 8.0);

    std::cout << "OK\n";
}

void test_matrix_scalar_zero() {
    std::cout << "[TEST] Matrix scalar multiplication by zero ... ";

    const Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const Matrix result = matrix * 0.0;

    assert(result(0, 0) == 0.0);
    assert(result(0, 1) == 0.0);
    assert(result(1, 0) == 0.0);
    assert(result(1, 1) == 0.0);

    std::cout << "OK\n";
}

void test_matrix_scalar_division() {
    std::cout << "[TEST] Matrix scalar division ... ";

    const Matrix matrix = {
        {2.0, 4.0},
        {6.0, 8.0}
    };

    const Matrix result = matrix / 2.0;

    assert(result(0, 0) == 1.0);
    assert(result(0, 1) == 2.0);
    assert(result(1, 0) == 3.0);
    assert(result(1, 1) == 4.0);

    std::cout << "OK\n";
}

void test_matrix_scalar_division_by_zero() {
    std::cout << "[TEST] Matrix scalar division by zero ... ";

    const Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    bool exception_thrown = false;

    try {
        [[maybe_unused]] const Matrix result = matrix / 0.0;
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);

    std::cout << "OK\n";
}

void test_matrix_inplace_scalar_multiplication() {
    std::cout << "[TEST] Matrix inplace scalar multiplication ... ";

    Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    matrix *= 3.0;

    assert(matrix(0, 0) == 3.0);
    assert(matrix(0, 1) == 6.0);
    assert(matrix(1, 0) == 9.0);
    assert(matrix(1, 1) == 12.0);

    std::cout << "OK\n";
}

void test_matrix_inplace_scalar_division() {
    std::cout << "[TEST] Matrix inplace scalar division ... ";

    Matrix matrix = {
        {2.0, 4.0},
        {6.0, 8.0}
    };

    matrix /= 2.0;

    assert(matrix(0, 0) == 1.0);
    assert(matrix(0, 1) == 2.0);
    assert(matrix(1, 0) == 3.0);
    assert(matrix(1, 1) == 4.0);

    std::cout << "OK\n";
}

} // namespace

int main() {
    std::cout << "Running matrix scalar tests...\n\n";

    test_matrix_scalar_multiplication();
    test_matrix_scalar_multiplication_negative();
    test_matrix_scalar_zero();
    test_matrix_scalar_division();
    test_matrix_scalar_division_by_zero();
    test_matrix_inplace_scalar_multiplication();
    test_matrix_inplace_scalar_division();

    std::cout << "\nAll matrix scalar tests passed.\n";

    return 0;
}