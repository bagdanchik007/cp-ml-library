#include "ml/ml.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace ml;

namespace {

void test_matrix_sum() {
    std::cout << "[TEST] Matrix sum ... ";

    const Matrix matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    assert(matrix.sum() == 45.0);

    std::cout << "OK\n";
}

void test_matrix_mean() {
    std::cout << "[TEST] Matrix mean ... ";

    const Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    assert(matrix.mean() == 2.5);

    std::cout << "OK\n";
}

void test_matrix_min() {
    std::cout << "[TEST] Matrix minimum ... ";

    const Matrix matrix = {
        {5.0, 3.0, 8.0},
        {1.0, 4.0, 6.0}
    };

    assert(matrix.min() == 1.0);

    std::cout << "OK\n";
}

void test_matrix_max() {
    std::cout << "[TEST] Matrix maximum ... ";

    const Matrix matrix = {
        {5.0, 3.0, 8.0},
        {1.0, 4.0, 6.0}
    };

    assert(matrix.max() == 8.0);

    std::cout << "OK\n";
}

void test_matrix_statistics_empty_matrix() {
    std::cout << "[TEST] Empty matrix statistics ... ";

    const Matrix matrix(0, 0);

    assert(matrix.sum() == 0.0);

    bool mean_thrown = false;

    try {
        [[maybe_unused]] const double result = matrix.mean();
    } catch (const std::invalid_argument&) {
        mean_thrown = true;
    }

    assert(mean_thrown);

    bool min_thrown = false;

    try {
        [[maybe_unused]] const double result = matrix.min();
    } catch (const std::invalid_argument&) {
        min_thrown = true;
    }

    assert(min_thrown);

    bool max_thrown = false;

    try {
        [[maybe_unused]] const double result = matrix.max();
    } catch (const std::invalid_argument&) {
        max_thrown = true;
    }

    assert(max_thrown);

    std::cout << "OK\n";
}

} // namespace

int main() {
    std::cout << "Running matrix statistics tests...\n\n";

    test_matrix_sum();
    test_matrix_mean();
    test_matrix_min();
    test_matrix_max();
    test_matrix_statistics_empty_matrix();

    std::cout << "\nAll matrix statistics tests passed.\n";

    return 0;
}