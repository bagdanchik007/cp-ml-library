#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "ml/core/matrix/matrix.hpp"

using ml::Matrix;

namespace {

void test_matrix_lu_decomposition() {
    std::cout << "[TEST] Matrix LU decomposition ... ";

    const Matrix matrix = {
        {4.0, 3.0},
        {6.0, 3.0}
    };

    const auto [lower, upper] =
        matrix.lu_decomposition();

    assert(std::abs(lower(0, 0) - 1.0) < 1e-12);
    assert(std::abs(lower(0, 1) - 0.0) < 1e-12);
    assert(std::abs(lower(1, 0) - 1.5) < 1e-12);
    assert(std::abs(lower(1, 1) - 1.0) < 1e-12);

    assert(std::abs(upper(0, 0) - 4.0) < 1e-12);
    assert(std::abs(upper(0, 1) - 3.0) < 1e-12);
    assert(std::abs(upper(1, 0) - 0.0) < 1e-12);
    assert(std::abs(upper(1, 1) + 1.5) < 1e-12);

    std::cout << "OK\n";
}

void test_matrix_lu_reconstructs_matrix() {
    std::cout << "[TEST] Matrix LU reconstruction ... ";

    const Matrix matrix = {
        {4.0, 3.0},
        {6.0, 3.0}
    };

    const auto [lower, upper] =
        matrix.lu_decomposition();

    const Matrix reconstructed =
        lower * upper;

    for (size_t i = 0; i < matrix.rows; ++i) {
        for (size_t j = 0; j < matrix.cols; ++j) {
            assert(
                std::abs(
                    reconstructed(i, j) - matrix(i, j)
                ) < 1e-12
            );
        }
    }

    std::cout << "OK\n";
}

void test_matrix_lu_rejects_non_square_matrix() {
    std::cout << "[TEST] Matrix LU rejects non-square matrix ... ";

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

void test_matrix_lu_rejects_empty_matrix() {
    std::cout << "[TEST] Matrix LU rejects empty matrix ... ";

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
void test_matrix_solve_linear_system() {
    std::cout << "[TEST] Matrix solves linear system ... ";

    const Matrix matrix = {
        {2.0, 1.0},
        {5.0, 7.0}
    };

    const std::vector<double> b = {
        11.0,
        13.0
    };

    const std::vector<double> solution =
        matrix.solve(b);

    assert(solution.size() == 2);

    assert(std::abs(solution[0] - 64.0 / 9.0) < 1e-12);
    assert(std::abs(solution[1] + 29.0 / 9.0) < 1e-12);

    std::cout << "OK\n";
}

void test_matrix_solve_rejects_non_square_matrix() {
    std::cout << "[TEST] Matrix solve rejects non-square matrix ... ";

    const Matrix matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    const std::vector<double> b = {
        1.0,
        2.0
    };

    bool exception_thrown = false;

    try {
        [[maybe_unused]] const auto solution =
            matrix.solve(b);
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);

    std::cout << "OK\n";
}

void test_matrix_solve_rejects_invalid_vector_size() {
    std::cout << "[TEST] Matrix solve rejects invalid vector size ... ";

    const Matrix matrix = {
        {2.0, 1.0},
        {5.0, 7.0}
    };

    const std::vector<double> b = {
        1.0,
        2.0,
        3.0
    };

    bool exception_thrown = false;

    try {
        [[maybe_unused]] const auto solution =
            matrix.solve(b);
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);

    std::cout << "OK\n";
}

void test_matrix_solve_rejects_empty_matrix() {
    std::cout << "[TEST] Matrix solve rejects empty matrix ... ";

    const Matrix matrix;

    const std::vector<double> b;

    bool exception_thrown = false;

    try {
        [[maybe_unused]] const auto solution =
            matrix.solve(b);
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);

    std::cout << "OK\n";
}

} // namespace

int main() {
    std::cout << "Running matrix decomposition tests...\n\n";

    test_matrix_lu_decomposition();
    test_matrix_lu_reconstructs_matrix();
    test_matrix_lu_rejects_non_square_matrix();
    test_matrix_lu_rejects_empty_matrix();

    test_matrix_solve_linear_system();
    test_matrix_solve_rejects_non_square_matrix();
    test_matrix_solve_rejects_invalid_vector_size();
    test_matrix_solve_rejects_empty_matrix();

    std::cout << "\nAll matrix decomposition tests passed.\n";

    return 0;
}