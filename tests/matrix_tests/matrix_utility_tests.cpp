#include "ml/ml.hpp"

#include <cassert>
#include <iostream>
#include <vector>

using namespace ml;

namespace {

void test_factory_methods() {
    std::cout << "[TEST] Matrix factory methods ... ";

    const Matrix zeros = Matrix::zeros(2, 3);

    assert(zeros.rows == 2);
    assert(zeros.cols == 3);

    for (double value : zeros.data) {
        assert(value == 0.0);
    }

    const Matrix ones = Matrix::ones(2, 3);

    assert(ones.rows == 2);
    assert(ones.cols == 3);

    for (double value : ones.data) {
        assert(value == 1.0);
    }

    std::cout << "OK\n";
}

void test_row_operations() {
    std::cout << "[TEST] Row operations ... ";

    Matrix m = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const auto row = m.row(1);

    assert(row.size() == 2);
    assert(row[0] == 3.0);
    assert(row[1] == 4.0);

    const std::vector<double> replacement = {
        10.0,
        20.0
    };

    m.set_row(0, replacement);

    assert(m(0, 0) == 10.0);
    assert(m(0, 1) == 20.0);

    std::cout << "OK\n";
}

void test_random_matrix() {
    std::cout << "[TEST] Random matrix ... ";

    const Matrix a =
        Matrix::random(3, 3, -1.0, 1.0, 42);

    const Matrix b =
        Matrix::random(3, 3, -1.0, 1.0, 42);

    assert(a.rows == 3);
    assert(a.cols == 3);

    assert(a.data == b.data);

    for (double value : a.data) {
        assert(value >= -1.0);
        assert(value <= 1.0);
    }

    std::cout << "OK\n";
}

void test_identity_matrix() {
    std::cout << "[TEST] Identity matrix ... ";

    const Matrix identity = Matrix::identity(3);

    assert(identity.rows == 3);
    assert(identity.cols == 3);

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            if (i == j) {
                assert(identity(i, j) == 1.0);
            } else {
                assert(identity(i, j) == 0.0);
            }
        }
    }

    std::cout << "OK\n";
}

void test_identity_matrix_multiplication() {
    std::cout << "[TEST] Identity matrix multiplication ... ";

    Matrix m = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const Matrix identity = Matrix::identity(2);

    const Matrix left_result = identity * m;
    const Matrix right_result = m * identity;

    assert(left_result.rows == m.rows);
    assert(left_result.cols == m.cols);

    assert(right_result.rows == m.rows);
    assert(right_result.cols == m.cols);

    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            assert(left_result(i, j) == m(i, j));
            assert(right_result(i, j) == m(i, j));
        }
    }

    std::cout << "OK\n";
}

} // namespace

int main() {
    std::cout << "Running matrix utility tests...\n\n";

    test_factory_methods();
    test_row_operations();
    test_random_matrix();
    test_identity_matrix();
    test_identity_matrix_multiplication();

    std::cout << "\nAll matrix utility tests passed!\n";

    return 0;
}