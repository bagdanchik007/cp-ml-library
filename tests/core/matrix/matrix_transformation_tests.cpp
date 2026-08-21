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

void test_matrix_submatrix() {
    std::cout << "[TEST] Matrix submatrix extraction ... ";

    const Matrix matrix = {
        {1.0, 2.0, 3.0, 4.0},
        {5.0, 6.0, 7.0, 8.0},
        {9.0, 10.0, 11.0, 12.0}
    };

    const Matrix result = matrix.submatrix(0, 2, 1, 3);

    assert(result.rows == 2);
    assert(result.cols == 2);

    assert(result(0, 0) == 2.0);
    assert(result(0, 1) == 3.0);
    assert(result(1, 0) == 6.0);
    assert(result(1, 1) == 7.0);

    std::cout << "OK\n";
}

void test_matrix_submatrix_rejects_invalid_range() {
    std::cout << "[TEST] Matrix submatrix rejects invalid range ... ";

    const Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    bool exception_thrown = false;

    try {
        [[maybe_unused]] const Matrix result =
            matrix.submatrix(1, 0, 0, 2);
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);

    std::cout << "OK\n";
}

void test_matrix_submatrix_rejects_out_of_bounds() {
    std::cout << "[TEST] Matrix submatrix rejects out of bounds ... ";

    const Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    bool exception_thrown = false;

    try {
        [[maybe_unused]] const Matrix result =
            matrix.submatrix(0, 3, 0, 2);
    } catch (const std::out_of_range&) {
        exception_thrown = true;
    }

    assert(exception_thrown);

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
    test_matrix_submatrix();
    test_matrix_submatrix_rejects_invalid_range();
    test_matrix_submatrix_rejects_out_of_bounds();   
    std::cout << "\nAll matrix transformation tests passed.\n";

    return 0;
}