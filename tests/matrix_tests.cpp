#include "ml/ml.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace ml;

void test_matrix() {
    std::cout << "[TEST] Matrix utilities ... ";

    Matrix m = {{1.0, 2.0}, {3.0, 4.0}};

    assert(m.rows == 2 && m.cols == 2);
    assert(m(0, 1) == 2.0);

    auto t = m.transpose();

    assert(t.rows == 2 && t.cols == 2);
    assert(t(1, 0) == 2.0);

    std::vector<double> v = {1.0, 1.0};
    auto res = m * v;

    assert(std::abs(res[0] - 3.0) < 1e-9);
    assert(std::abs(res[1] - 7.0) < 1e-9);

    std::cout << "OK\n";
}

void test_matrix_error_handling() {
    std::cout << "[TEST] Matrix error handling ... ";

    bool threw = false;

    // Addition with incompatible dimensions
    try {
        Matrix a(2, 2);
        Matrix b(3, 3);
        [[maybe_unused]] auto result = a + b;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    // Subtraction with incompatible dimensions
    threw = false;

    try {
        Matrix a(2, 2);
        Matrix b(2, 3);
        [[maybe_unused]] auto result = a - b;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    // Matrix multiplication with incompatible dimensions
    threw = false;

    try {
        Matrix a(2, 3);
        Matrix b(2, 2);
        [[maybe_unused]] auto result = a * b;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    // Matrix-vector multiplication with incompatible dimensions
    threw = false;

    try {
        Matrix a(2, 3);
        std::vector<double> vector = {1.0, 2.0};
        [[maybe_unused]] auto result = a * vector;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    // Division by zero
    threw = false;

    try {
        Matrix a(2, 2, 1.0);
        [[maybe_unused]] auto result = a / 0.0;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    std::cout << "OK\n";
}

int main() {
    test_matrix();
    test_matrix_error_handling();
    std::cout << "\nAll matrix tests passed!\n";
    return 0;
}