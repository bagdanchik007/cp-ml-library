#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "ml/core/matrix/matrix.hpp"

using ml::Matrix;

namespace {

void test_matrix_solve_1x1() {
    std::cout << "[TEST] Matrix solve 1x1 system ... ";

    const Matrix matrix = {
        {4.0}
    };

    const std::vector<double> b = {
        12.0
    };

    const std::vector<double> solution =
        matrix.solve(b);

    assert(solution.size() == 1);
    assert(std::abs(solution[0] - 3.0) < 1e-12);

    std::cout << "OK\n";
}

void test_matrix_solve_2x2() {
    std::cout << "[TEST] Matrix solve 2x2 system ... ";

    const Matrix matrix = {
        {2.0, 1.0},
        {1.0, 3.0}
    };

    const std::vector<double> b = {
        5.0,
        6.0
    };

    const std::vector<double> solution =
        matrix.solve(b);

    assert(solution.size() == 2);

    assert(std::abs(solution[0] - 1.8) < 1e-12);
    assert(std::abs(solution[1] - 1.4) < 1e-12);

    std::cout << "OK\n";
}

void test_matrix_solve_3x3() {
    std::cout << "[TEST] Matrix solve 3x3 system ... ";

    const Matrix matrix = {
        {3.0, 2.0, -1.0},
        {2.0, -2.0, 4.0},
        {-1.0, 0.5, -1.0}
    };

    const std::vector<double> b = {
        1.0,
        -2.0,
        0.0
    };

    const std::vector<double> solution =
        matrix.solve(b);

    assert(solution.size() == 3);

    assert(std::abs(solution[0] - 1.0) < 1e-12);
    assert(std::abs(solution[1] - (-2.0)) < 1e-12);
    assert(std::abs(solution[2] - (-2.0)) < 1e-12);

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

void test_matrix_solve_rejects_invalid_rhs_size() {
    std::cout << "[TEST] Matrix solve rejects invalid RHS size ... ";

    const Matrix matrix = {
        {2.0, 1.0},
        {1.0, 3.0}
    };

    const std::vector<double> b = {
        1.0
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

void test_matrix_solve_rejects_singular_matrix() {
    std::cout << "[TEST] Matrix solve rejects singular matrix ... ";

    const Matrix matrix = {
        {1.0, 2.0},
        {2.0, 4.0}
    };

    const std::vector<double> b = {
        3.0,
        6.0
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

} // namespace

int main() {
    std::cout << "Running matrix solve tests...\n\n";

    test_matrix_solve_1x1();
    test_matrix_solve_2x2();
    test_matrix_solve_3x3();

    test_matrix_solve_rejects_non_square_matrix();
    test_matrix_solve_rejects_invalid_rhs_size();
    test_matrix_solve_rejects_empty_matrix();
    test_matrix_solve_rejects_singular_matrix();

    std::cout << "\nAll matrix solve tests passed.\n";

    return 0;
}