#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "ml/core/matrix/matrix.hpp"

using ml::Matrix;

namespace {

constexpr double EPSILON = 1e-12;

void assert_matrix_equal(
    const Matrix& lhs,
    const Matrix& rhs
) {
    assert(lhs.rows == rhs.rows);
    assert(lhs.cols == rhs.cols);

    for (size_t i = 0; i < lhs.rows; ++i) {
        for (size_t j = 0; j < lhs.cols; ++j) {
            assert(
                std::abs(lhs(i, j) - rhs(i, j))
                < EPSILON
            );
        }
    }
}

// ============================================================
// LU decomposition
// ============================================================

void test_matrix_lu_decomposition() {
    std::cout << "[TEST] Matrix LU decomposition ... ";

    const Matrix matrix = {
        {4.0, 3.0},
        {6.0, 3.0}
    };

    const auto decomposition =
        matrix.lu_decomposition();

    const Matrix& lower =
        decomposition.lower;

    const Matrix& upper =
        decomposition.upper;

    const Matrix& permutation =
        decomposition.permutation;

    assert(std::abs(lower(0, 0) - 1.0) < EPSILON);
    assert(std::abs(lower(0, 1) - 0.0) < EPSILON);
    assert(std::abs(lower(1, 0) - 2.0 / 3.0) < EPSILON);
    assert(std::abs(lower(1, 1) - 1.0) < EPSILON);

    assert(std::abs(upper(0, 0) - 6.0) < EPSILON);
    assert(std::abs(upper(0, 1) - 3.0) < EPSILON);
    assert(std::abs(upper(1, 0) - 0.0) < EPSILON);
    assert(std::abs(upper(1, 1) - 1.0) < EPSILON);

    const Matrix expected_permutation = {
        {0.0, 1.0},
        {1.0, 0.0}
    };

    assert_matrix_equal(
        permutation,
        expected_permutation
    );

    std::cout << "OK\n";
}

// ============================================================
// LU reconstruction
// ============================================================

void test_matrix_lu_reconstructs_matrix() {
    std::cout << "[TEST] Matrix LU reconstruction ... ";

    const Matrix matrix = {
        {4.0, 3.0},
        {6.0, 3.0}
    };

    const auto decomposition =
        matrix.lu_decomposition();

    const Matrix& lower =
        decomposition.lower;

    const Matrix& upper =
        decomposition.upper;

    const Matrix& permutation =
        decomposition.permutation;

    const Matrix permuted =
        permutation * matrix;

    const Matrix reconstructed =
        lower * upper;

    assert_matrix_equal(
        permuted,
        reconstructed
    );

    std::cout << "OK\n";
}

// ============================================================
// LU partial pivoting
// ============================================================

void test_matrix_lu_partial_pivoting() {
    std::cout << "[TEST] Matrix LU partial pivoting ... ";

    const Matrix matrix = {
        {0.0, 2.0},
        {1.0, 3.0}
    };

    const auto decomposition =
        matrix.lu_decomposition();

    const Matrix& lower =
        decomposition.lower;

    const Matrix& upper =
        decomposition.upper;

    const Matrix& permutation =
        decomposition.permutation;

    const Matrix permuted =
        permutation * matrix;

    const Matrix reconstructed =
        lower * upper;

    assert_matrix_equal(
        permuted,
        reconstructed
    );

    const Matrix expected_permutation = {
        {0.0, 1.0},
        {1.0, 0.0}
    };

    assert_matrix_equal(
        permutation,
        expected_permutation
    );

    assert(std::abs(lower(0, 0) - 1.0) < EPSILON);
    assert(std::abs(lower(0, 1) - 0.0) < EPSILON);
    assert(std::abs(lower(1, 0) - 0.0) < EPSILON);
    assert(std::abs(lower(1, 1) - 1.0) < EPSILON);

    assert(std::abs(upper(0, 0) - 1.0) < EPSILON);
    assert(std::abs(upper(0, 1) - 3.0) < EPSILON);
    assert(std::abs(upper(1, 0) - 0.0) < EPSILON);
    assert(std::abs(upper(1, 1) - 2.0) < EPSILON);

    std::cout << "OK\n";
}

// ============================================================
// Non-square matrix
// ============================================================

void test_matrix_lu_rejects_non_square_matrix() {
    std::cout
        << "[TEST] Matrix LU rejects non-square matrix ... ";

    const Matrix matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    bool exception_thrown = false;

    try {
        [[maybe_unused]] const auto result =
            matrix.lu_decomposition();
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);

    std::cout << "OK\n";
}

// ============================================================
// Empty matrix
// ============================================================

void test_matrix_lu_rejects_empty_matrix() {
    std::cout
        << "[TEST] Matrix LU rejects empty matrix ... ";

    const Matrix matrix;

    bool exception_thrown = false;

    try {
        [[maybe_unused]] const auto result =
            matrix.lu_decomposition();
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);

    std::cout << "OK\n";
}

// ============================================================
// Singular matrix
// ============================================================

void test_matrix_lu_rejects_singular_matrix() {
    std::cout
        << "[TEST] Matrix LU rejects singular matrix ... ";

    const Matrix matrix = {
        {1.0, 2.0},
        {2.0, 4.0}
    };

    bool exception_thrown = false;

    try {
        [[maybe_unused]] const auto result =
            matrix.lu_decomposition();
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);

    std::cout << "OK\n";
}

// ============================================================
// Linear system solve
// ============================================================

void test_matrix_solve() {
    std::cout << "[TEST] Matrix solve ... ";

    const Matrix matrix = {
        {3.0, 2.0},
        {1.0, 2.0}
    };

    const std::vector<double> b = {
        7.0,
        5.0
    };

    const std::vector<double> x =
        matrix.solve(b);

    assert(std::abs(x[0] - 1.0) < EPSILON);
    assert(std::abs(x[1] - 2.0) < EPSILON);

    std::cout << "OK\n";
}

// ============================================================
// Solve with pivoting
// ============================================================

void test_matrix_solve_with_pivoting() {
    std::cout
        << "[TEST] Matrix solve with pivoting ... ";

    const Matrix matrix = {
        {0.0, 2.0},
        {1.0, 3.0}
    };

    const std::vector<double> b = {
        4.0,
        5.0
    };

    const std::vector<double> x =
        matrix.solve(b);

    assert(std::abs(x[0] - (-1.0)) < EPSILON);
    assert(std::abs(x[1] - 2.0) < EPSILON);

    std::cout << "OK\n";
}

// ============================================================
// Solve invalid RHS size
// ============================================================

void test_matrix_solve_rejects_invalid_rhs_size() {
    std::cout
        << "[TEST] Matrix solve rejects invalid RHS size ... ";

    const Matrix matrix = {
        {2.0, 1.0},
        {1.0, 2.0}
    };

    const std::vector<double> b = {
        1.0
    };

    bool exception_thrown = false;

    try {
        [[maybe_unused]] const auto result =
            matrix.solve(b);
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);

    std::cout << "OK\n";
}

} // namespace

int main() {
    std::cout
        << "Running matrix decomposition tests...\n\n";

    test_matrix_lu_decomposition();
    test_matrix_lu_reconstructs_matrix();
    test_matrix_lu_partial_pivoting();
    test_matrix_lu_rejects_non_square_matrix();
    test_matrix_lu_rejects_empty_matrix();
    test_matrix_lu_rejects_singular_matrix();

    test_matrix_solve();
    test_matrix_solve_with_pivoting();
    test_matrix_solve_rejects_invalid_rhs_size();

    std::cout
        << "\nAll matrix decomposition tests passed.\n";

    return 0;
}