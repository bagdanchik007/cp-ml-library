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

void test_matrix_determinant() {
    std::cout << "[TEST] Matrix determinant ... ";

    const Matrix matrix_2x2 = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    assert(matrix_2x2.determinant() == -2.0);

    const Matrix matrix_3x3 = {
        {1.0, 2.0, 3.0},
        {0.0, 1.0, 4.0},
        {5.0, 6.0, 0.0}
    };

    assert(matrix_3x3.determinant() == 1.0);

    const Matrix matrix_1x1 = {
        {7.0}
    };

    assert(matrix_1x1.determinant() == 7.0);

    std::cout << "OK\n";
}

void test_matrix_inverse() {
    std::cout << "[TEST] Matrix inverse ... ";

    const Matrix matrix = {
        {4.0, 7.0},
        {2.0, 6.0}
    };

    const Matrix inverse = matrix.inverse();

    assert(std::abs(inverse(0, 0) - 0.6) < 1e-9);
    assert(std::abs(inverse(0, 1) - (-0.7)) < 1e-9);
    assert(std::abs(inverse(1, 0) - (-0.2)) < 1e-9);
    assert(std::abs(inverse(1, 1) - 0.4) < 1e-9);

    const Matrix identity = matrix * inverse;

    assert(std::abs(identity(0, 0) - 1.0) < 1e-9);
    assert(std::abs(identity(0, 1)) < 1e-9);
    assert(std::abs(identity(1, 0)) < 1e-9);
    assert(std::abs(identity(1, 1) - 1.0) < 1e-9);

    std::cout << "OK\n";
}

void test_euclidean_distance() {
    std::cout << "[TEST] Euclidean distance ... ";

    const std::vector<double> a = {
        0.0,
        0.0
    };

    const std::vector<double> b = {
        3.0,
        4.0
    };

    const double distance = euclidean_distance(a, b);

    assert(std::abs(distance - 5.0) < 1e-9);

    const std::vector<double> c = {
        1.0,
        2.0,
        3.0
    };

    const std::vector<double> d = {
        1.0,
        2.0,
        3.0
    };

    assert(std::abs(
        euclidean_distance(c, d)
    ) < 1e-9);

    std::cout << "OK\n";
}

void test_matrix_trace() {
    std::cout << "[TEST] Matrix trace ... ";

    const Matrix matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    assert(matrix.trace() == 15.0);

    const Matrix matrix_2x2 = {
        {4.0, 2.0},
        {1.0, 7.0}
    };

    assert(matrix_2x2.trace() == 11.0);

    const Matrix matrix_1x1 = {
        {42.0}
    };

    assert(matrix_1x1.trace() == 42.0);

    bool exception_thrown = false;

    const Matrix non_square = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    try {
        [[maybe_unused]] const double result =
            non_square.trace();
    } catch (const std::invalid_argument&) {
        exception_thrown = true;
    }

    assert(exception_thrown);

    std::cout << "OK\n";
}

void test_squared_norm() {
    ml::Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    assert(matrix.squared_norm() == 30.0);
}

void test_norm() {
    ml::Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    assert(
        std::abs(matrix.norm() - std::sqrt(30.0)) < 1e-9
    );
}

} // namespace

int main() {
    std::cout << "Running matrix utility tests...\n\n";

    test_factory_methods();
    test_row_operations();
    test_random_matrix();
    test_identity_matrix();
    test_identity_matrix_multiplication();
    test_matrix_determinant();
    test_matrix_inverse();
    test_euclidean_distance();
    test_matrix_trace();
    test_squared_norm();
    test_norm();
    std::cout << "\nAll matrix utility tests passed!\n";

    return 0;
}