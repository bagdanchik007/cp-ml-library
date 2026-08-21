#include "ml/ml.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ml;

namespace {

// ============================================================
// Reshape tests
// ============================================================

void test_matrix_reshape_changes_dimensions() {
    std::cout << "[TEST] Matrix reshape changes dimensions ... ";

    Matrix matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    matrix.reshape(3, 2);

    assert(matrix.rows == 3);
    assert(matrix.cols == 2);

    std::cout << "OK\n";
}

void test_matrix_reshape_preserves_data_order() {
    std::cout << "[TEST] Matrix reshape preserves data order ... ";

    Matrix matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    matrix.reshape(3, 2);

    assert(matrix(0, 0) == 1.0);
    assert(matrix(0, 1) == 2.0);
    assert(matrix(1, 0) == 3.0);
    assert(matrix(1, 1) == 4.0);
    assert(matrix(2, 0) == 5.0);
    assert(matrix(2, 1) == 6.0);

    std::cout << "OK\n";
}

void test_matrix_reshape_rejects_invalid_dimensions() {
    std::cout << "[TEST] Matrix reshape rejects invalid dimensions ... ";

    Matrix matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    bool exception_thrown = false;

    try {
        matrix.reshape(3, 3);
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);

    std::cout << "OK\n";
}

// ============================================================
// Flatten tests
// ============================================================

void test_matrix_flatten_returns_row_major_order() {
    std::cout << "[TEST] Matrix flatten returns row-major order ... ";

    const Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const std::vector<double> values = matrix.flatten();

    assert(values.size() == 4);

    assert(values[0] == 1.0);
    assert(values[1] == 2.0);
    assert(values[2] == 3.0);
    assert(values[3] == 4.0);

    std::cout << "OK\n";
}

void test_matrix_flatten_does_not_modify_matrix() {
    std::cout << "[TEST] Matrix flatten does not modify matrix ... ";

    const Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const auto values = matrix.flatten();

    assert(values.size() == 4);

    assert(matrix.rows == 2);
    assert(matrix.cols == 2);

    assert(matrix(0, 0) == 1.0);
    assert(matrix(0, 1) == 2.0);
    assert(matrix(1, 0) == 3.0);
    assert(matrix(1, 1) == 4.0);

    std::cout << "OK\n";
}

// ============================================================
// Edge cases
// ============================================================

void test_matrix_reshape_to_single_row() {
    std::cout << "[TEST] Matrix reshape to single row ... ";

    Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    matrix.reshape(1, 4);

    assert(matrix.rows == 1);
    assert(matrix.cols == 4);

    assert(matrix(0, 0) == 1.0);
    assert(matrix(0, 1) == 2.0);
    assert(matrix(0, 2) == 3.0);
    assert(matrix(0, 3) == 4.0);

    std::cout << "OK\n";
}

void test_matrix_reshape_to_single_column() {
    std::cout << "[TEST] Matrix reshape to single column ... ";

    Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    matrix.reshape(4, 1);

    assert(matrix.rows == 4);
    assert(matrix.cols == 1);

    assert(matrix(0, 0) == 1.0);
    assert(matrix(1, 0) == 2.0);
    assert(matrix(2, 0) == 3.0);
    assert(matrix(3, 0) == 4.0);

    std::cout << "OK\n";
}

void test_matrix_flatten_empty_matrix() {
    std::cout << "[TEST] Matrix flatten empty matrix ... ";

    const Matrix matrix(0, 0);

    const auto values = matrix.flatten();

    assert(values.empty());

    std::cout << "OK\n";
}

} // namespace

int main() {
    std::cout << "Running matrix transformation tests...\n\n";

    test_matrix_reshape_changes_dimensions();
    test_matrix_reshape_preserves_data_order();
    test_matrix_reshape_rejects_invalid_dimensions();

    test_matrix_flatten_returns_row_major_order();
    test_matrix_flatten_does_not_modify_matrix();

    test_matrix_reshape_to_single_row();
    test_matrix_reshape_to_single_column();
    test_matrix_flatten_empty_matrix();

    std::cout << "\nAll matrix transformation tests passed.\n";

    return 0;
}