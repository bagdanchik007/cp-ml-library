#include "ml/ml.hpp"

#include <cassert>
#include <iostream>

using namespace ml;

namespace {

void test_matrix_addition() {
    std::cout << "[TEST] Matrix addition ... ";

    Matrix a = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    Matrix b = {
        {5.0, 6.0},
        {7.0, 8.0}
    };

    const Matrix result = a + b;

    assert(result.rows == 2);
    assert(result.cols == 2);

    assert(result(0, 0) == 6.0);
    assert(result(0, 1) == 8.0);
    assert(result(1, 0) == 10.0);
    assert(result(1, 1) == 12.0);

    std::cout << "OK\n";
}

void test_matrix_subtraction() {
    std::cout << "[TEST] Matrix subtraction ... ";

    Matrix a = {
        {5.0, 6.0},
        {7.0, 8.0}
    };

    Matrix b = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const Matrix result = a - b;

    assert(result(0, 0) == 4.0);
    assert(result(0, 1) == 4.0);
    assert(result(1, 0) == 4.0);
    assert(result(1, 1) == 4.0);

    std::cout << "OK\n";
}

void test_matrix_multiplication() {
    std::cout << "[TEST] Matrix multiplication ... ";

    Matrix a = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    Matrix b = {
        {7.0, 8.0},
        {9.0, 10.0},
        {11.0, 12.0}
    };

    const Matrix result = a * b;

    assert(result.rows == 2);
    assert(result.cols == 2);

    assert(result(0, 0) == 58.0);
    assert(result(0, 1) == 64.0);
    assert(result(1, 0) == 139.0);
    assert(result(1, 1) == 154.0);

    std::cout << "OK\n";
}

void test_inplace_addition() {
    std::cout << "[TEST] In-place matrix addition ... ";

    Matrix a = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    Matrix b = {
        {5.0, 6.0},
        {7.0, 8.0}
    };

    a += b;

    assert(a(0, 0) == 6.0);
    assert(a(0, 1) == 8.0);
    assert(a(1, 0) == 10.0);
    assert(a(1, 1) == 12.0);

    std::cout << "OK\n";
}

void test_inplace_subtraction() {
    std::cout << "[TEST] In-place matrix subtraction ... ";

    Matrix a = {
        {6.0, 8.0},
        {10.0, 12.0}
    };

    Matrix b = {
        {5.0, 6.0},
        {7.0, 8.0}
    };

    a -= b;

    assert(a(0, 0) == 1.0);
    assert(a(0, 1) == 2.0);
    assert(a(1, 0) == 3.0);
    assert(a(1, 1) == 4.0);

    std::cout << "OK\n";
}

void test_scalar_operations() {
    std::cout << "[TEST] Scalar operations ... ";

    Matrix m = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const Matrix multiplied = m * 2.0;

    assert(multiplied(0, 0) == 2.0);
    assert(multiplied(0, 1) == 4.0);
    assert(multiplied(1, 0) == 6.0);
    assert(multiplied(1, 1) == 8.0);

    const Matrix divided = m / 2.0;

    assert(divided(0, 0) == 0.5);
    assert(divided(0, 1) == 1.0);
    assert(divided(1, 0) == 1.5);
    assert(divided(1, 1) == 2.0);

    std::cout << "OK\n";
}

void test_inplace_scalar_operations() {
    std::cout << "[TEST] In-place scalar operations ... ";

    Matrix m = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    m *= 2.0;

    assert(m(0, 0) == 2.0);
    assert(m(0, 1) == 4.0);
    assert(m(1, 0) == 6.0);
    assert(m(1, 1) == 8.0);

    m /= 2.0;

    assert(m(0, 0) == 1.0);
    assert(m(0, 1) == 2.0);
    assert(m(1, 0) == 3.0);
    assert(m(1, 1) == 4.0);

    std::cout << "OK\n";
}

} // namespace

int main() {
    std::cout << "Running matrix arithmetic tests...\n\n";

    test_matrix_addition();
    test_matrix_subtraction();
    test_matrix_multiplication();

    test_inplace_addition();
    test_inplace_subtraction();

    test_scalar_operations();
    test_inplace_scalar_operations();

    std::cout << "\nAll matrix arithmetic tests passed!\n";

    return 0;
}