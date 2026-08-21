#include "ml/ml.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ml;

namespace {

void test_matrix_row() {
    std::cout << "[TEST] Matrix row extraction ... ";

    const Matrix matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    const std::vector<double> row = matrix.row(1);

    assert(row.size() == 3);
    assert(row[0] == 4.0);
    assert(row[1] == 5.0);
    assert(row[2] == 6.0);

    std::cout << "OK\n";
}

void test_matrix_set_row() {
    std::cout << "[TEST] Matrix set row ... ";

    Matrix matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    matrix.set_row(1, {10.0, 20.0, 30.0});

    assert(matrix(1, 0) == 10.0);
    assert(matrix(1, 1) == 20.0);
    assert(matrix(1, 2) == 30.0);

    std::cout << "OK\n";
}

void test_matrix_swap_rows() {
    std::cout << "[TEST] Matrix swap rows ... ";

    Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}
    };

    matrix.swap_rows(0, 2);

    assert(matrix(0, 0) == 5.0);
    assert(matrix(0, 1) == 6.0);
    assert(matrix(2, 0) == 1.0);
    assert(matrix(2, 1) == 2.0);

    std::cout << "OK\n";
}

void test_matrix_row_validation() {
    std::cout << "[TEST] Matrix row validation ... ";

    const Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    bool row_thrown = false;

    try {
        [[maybe_unused]] const auto result = matrix.row(2);
    } catch (const std::out_of_range&) {
        row_thrown = true;
    }

    assert(row_thrown);

    std::cout << "OK\n";
}

void test_matrix_set_row_invalid_size() {
    std::cout << "[TEST] Matrix set row rejects invalid size ... ";

    Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    bool exception_thrown = false;

    try {
        matrix.set_row(0, {10.0});
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);

    std::cout << "OK\n";
}

} // namespace

int main() {
    std::cout << "Running matrix row tests...\n\n";

    test_matrix_row();
    test_matrix_set_row();
    test_matrix_swap_rows();
    test_matrix_row_validation();
    test_matrix_set_row_invalid_size();

    std::cout << "\nAll matrix row tests passed.\n";

    return 0;
}