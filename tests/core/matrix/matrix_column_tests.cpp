#include "ml/ml.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ml;

namespace {

void test_matrix_column() {
    std::cout << "[TEST] Matrix column extraction ... ";

    const Matrix matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    const std::vector<double> column = matrix.column(1);

    assert(column.size() == 3);
    assert(column[0] == 2.0);
    assert(column[1] == 5.0);
    assert(column[2] == 8.0);

    std::cout << "OK\n";
}

void test_matrix_set_column() {
    std::cout << "[TEST] Matrix set column ... ";

    Matrix matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    matrix.set_column(1, {10.0, 20.0, 30.0});

    assert(matrix(0, 1) == 10.0);
    assert(matrix(1, 1) == 20.0);
    assert(matrix(2, 1) == 30.0);

    std::cout << "OK\n";
}

void test_matrix_swap_columns() {
    std::cout << "[TEST] Matrix swap columns ... ";

    Matrix matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    matrix.swap_columns(0, 2);

    assert(matrix(0, 0) == 3.0);
    assert(matrix(0, 2) == 1.0);
    assert(matrix(1, 0) == 6.0);
    assert(matrix(1, 2) == 4.0);

    std::cout << "OK\n";
}

void test_matrix_column_validation() {
    std::cout << "[TEST] Matrix column validation ... ";

    const Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    bool column_thrown = false;

    try {
        [[maybe_unused]] const auto result = matrix.column(2);
    } catch (const std::out_of_range&) {
        column_thrown = true;
    }

    assert(column_thrown);

    std::cout << "OK\n";
}

} // namespace

int main() {
    std::cout << "Running matrix column tests...\n\n";

    test_matrix_column();
    test_matrix_set_column();
    test_matrix_swap_columns();
    test_matrix_column_validation();

    std::cout << "\nAll matrix column tests passed.\n";

    return 0;
}