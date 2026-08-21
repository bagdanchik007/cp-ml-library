#include "ml/ml.hpp"

#include <cassert>
#include <iostream>

using namespace ml;

namespace {

void test_matrix_equal() {
    std::cout << "[TEST] Matrix equality ... ";

    const Matrix a = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const Matrix b = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    assert(a == b);

    std::cout << "OK\n";
}

void test_matrix_not_equal_values() {
    std::cout << "[TEST] Matrix inequality by values ... ";

    const Matrix a = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const Matrix b = {
        {1.0, 2.0},
        {3.0, 5.0}
    };

    assert(a != b);

    std::cout << "OK\n";
}

void test_matrix_not_equal_dimensions() {
    std::cout << "[TEST] Matrix inequality by dimensions ... ";

    const Matrix a = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const Matrix b = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    assert(a != b);

    std::cout << "OK\n";
}

void test_matrix_self_equality() {
    std::cout << "[TEST] Matrix self equality ... ";

    const Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    assert(matrix == matrix);
    assert(!(matrix != matrix));

    std::cout << "OK\n";
}

} // namespace

int main() {
    std::cout << "Running matrix comparison tests...\n\n";

    test_matrix_equal();
    test_matrix_not_equal_values();
    test_matrix_not_equal_dimensions();
    test_matrix_self_equality();

    std::cout << "\nAll matrix comparison tests passed.\n";

    return 0;
}