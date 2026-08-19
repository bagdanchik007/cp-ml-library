#include "ml/ml.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ml;

namespace {

void test_matrix_construction() {
    std::cout << "[TEST] Matrix construction ... ";

    Matrix m = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    assert(m.rows == 2);
    assert(m.cols == 2);

    assert(m(0, 0) == 1.0);
    assert(m(0, 1) == 2.0);
    assert(m(1, 0) == 3.0);
    assert(m(1, 1) == 4.0);

    std::cout << "OK\n";
}

void test_matrix_transpose() {
    std::cout << "[TEST] Matrix transpose ... ";

    Matrix m = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    const Matrix t = m.transpose();

    assert(t.rows == 3);
    assert(t.cols == 2);

    assert(t(0, 0) == 1.0);
    assert(t(0, 1) == 4.0);

    assert(t(1, 0) == 2.0);
    assert(t(1, 1) == 5.0);

    assert(t(2, 0) == 3.0);
    assert(t(2, 1) == 6.0);

    std::cout << "OK\n";
}

void test_matrix_vector_multiplication() {
    std::cout << "[TEST] Matrix-vector multiplication ... ";

    Matrix m = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const std::vector<double> v = {
        1.0,
        1.0
    };

    const auto result = m * v;

    assert(result.size() == 2);
    assert(std::abs(result[0] - 3.0) < 1e-9);
    assert(std::abs(result[1] - 7.0) < 1e-9);

    std::cout << "OK\n";
}

void test_matrix_addition() {
    std::cout << "[TEST] Matrix addition ... ";

    Matrix a = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    Matrix b = {
        {5.0, 6.0},
        {7.0, 8.0}
    };

    const Matrix result = a + b;

    assert(result.rows == 2);
    assert(result.cols == 2);

    assert(result(0, 0) == 6.0);
    assert(result(0, 1) == 8.0);
    assert(result(1, 0) == 10.0);
    assert(result(1, 1) == 12.0);

    std::cout << "OK\n";
}

void test_matrix_subtraction() {
    std::cout << "[TEST] Matrix subtraction ... ";

    Matrix a = {
        {5.0, 6.0},
        {7.0, 8.0}
    };

    Matrix b = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const Matrix result = a - b;

    assert(result(0, 0) == 4.0);
    assert(result(0, 1) == 4.0);
    assert(result(1, 0) == 4.0);
    assert(result(1, 1) == 4.0);

    std::cout << "OK\n";
}

void test_matrix_multiplication() {
    std::cout << "[TEST] Matrix multiplication ... ";

    Matrix a = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    Matrix b = {
        {7.0, 8.0},
        {9.0, 10.0},
        {11.0, 12.0}
    };

    const Matrix result = a * b;

    assert(result.rows == 2);
    assert(result.cols == 2);

    assert(result(0, 0) == 58.0);
    assert(result(0, 1) == 64.0);
    assert(result(1, 0) == 139.0);
    assert(result(1, 1) == 154.0);

    std::cout << "OK\n";
}

void test_inplace_addition() {
    std::cout << "[TEST] In-place matrix addition ... ";

    Matrix a = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    Matrix b = {
        {5.0, 6.0},
        {7.0, 8.0}
    };

    a += b;

    assert(a(0, 0) == 6.0);
    assert(a(0, 1) == 8.0);
    assert(a(1, 0) == 10.0);
    assert(a(1, 1) == 12.0);

    std::cout << "OK\n";
}

void test_inplace_subtraction() {
    std::cout << "[TEST] In-place matrix subtraction ... ";

    Matrix a = {
        {6.0, 8.0},
        {10.0, 12.0}
    };

    Matrix b = {
        {5.0, 6.0},
        {7.0, 8.0}
    };

    a -= b;

    assert(a(0, 0) == 1.0);
    assert(a(0, 1) == 2.0);
    assert(a(1, 0) == 3.0);
    assert(a(1, 1) == 4.0);

    std::cout << "OK\n";
}

void test_scalar_operations() {
    std::cout << "[TEST] Scalar operations ... ";

    Matrix m = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const Matrix multiplied = m * 2.0;

    assert(multiplied(0, 0) == 2.0);
    assert(multiplied(0, 1) == 4.0);
    assert(multiplied(1, 0) == 6.0);
    assert(multiplied(1, 1) == 8.0);

    const Matrix divided = m / 2.0;

    assert(divided(0, 0) == 0.5);
    assert(divided(0, 1) == 1.0);
    assert(divided(1, 0) == 1.5);
    assert(divided(1, 1) == 2.0);

    std::cout << "OK\n";
}

void test_inplace_scalar_operations() {
    std::cout << "[TEST] In-place scalar operations ... ";

    Matrix m = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    m *= 2.0;

    assert(m(0, 0) == 2.0);
    assert(m(0, 1) == 4.0);
    assert(m(1, 0) == 6.0);
    assert(m(1, 1) == 8.0);

    m /= 2.0;

    assert(m(0, 0) == 1.0);
    assert(m(0, 1) == 2.0);
    assert(m(1, 0) == 3.0);
    assert(m(1, 1) == 4.0);

    std::cout << "OK\n";
}

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

void test_dimension_errors() {
    std::cout << "[TEST] Matrix dimension validation ... ";

    Matrix a = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    Matrix incompatible = {
        {1.0, 2.0, 3.0}
    };

    bool addition_thrown = false;

    try {
        [[maybe_unused]] const Matrix result =
            a + incompatible;
    } catch (const std::invalid_argument&) {
        addition_thrown = true;
    }

    assert(addition_thrown);

    bool subtraction_thrown = false;

    try {
        [[maybe_unused]] const Matrix result =
            a - incompatible;
    } catch (const std::invalid_argument&) {
        subtraction_thrown = true;
    }

    assert(subtraction_thrown);

    Matrix invalid_multiplication = {
        {1.0, 2.0, 3.0}
    };

    bool multiplication_thrown = false;

    try {
        [[maybe_unused]] const Matrix result =
            a * invalid_multiplication;
    } catch (const std::invalid_argument&) {
        multiplication_thrown = true;
    }

    assert(multiplication_thrown);

    bool vector_thrown = false;

    try {
        const std::vector<double> invalid_vector = {
            1.0,
            2.0,
            3.0
        };

        [[maybe_unused]] const auto result =
            a * invalid_vector;
    } catch (const std::invalid_argument&) {
        vector_thrown = true;
    }

    assert(vector_thrown);

    std::cout << "OK\n";
}

void test_scalar_division_by_zero() {
    std::cout << "[TEST] Scalar division by zero ... ";

    Matrix m = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    bool division_thrown = false;

    try {
        [[maybe_unused]] const Matrix result =
            m / 0.0;
    } catch (const std::invalid_argument&) {
        division_thrown = true;
    }

    assert(division_thrown);

    bool inplace_division_thrown = false;

    try {
        m /= 0.0;
    } catch (const std::invalid_argument&) {
        inplace_division_thrown = true;
    }

    assert(inplace_division_thrown);

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

    const double distance =
        euclidean_distance(a, b);

    assert(std::abs(distance - 5.0) < 1e-9);

    std::cout << "OK\n";
}

void test_euclidean_distance_dimension_error() {
    std::cout << "[TEST] Euclidean distance validation ... ";

    const std::vector<double> a = {
        1.0,
        2.0
    };

    const std::vector<double> b = {
        1.0,
        2.0,
        3.0
    };

    bool thrown = false;

    try {
        [[maybe_unused]] const double result =
            euclidean_distance(a, b);
    } catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);

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

void test_determinant_1x1() {
    std::cout << "[TEST] Determinant 1x1 ... ";

    Matrix m = {
        {5.0}
    };

    const double result = m.determinant();

    assert(std::abs(result - 5.0) < 1e-9);

    std::cout << "OK\n";
}

void test_determinant_2x2() {
    std::cout << "[TEST] Determinant 2x2 ... ";

    Matrix m = {
        {4.0, 7.0},
        {2.0, 6.0}
    };

    const double result = m.determinant();

    assert(std::abs(result - 10.0) < 1e-9);

    std::cout << "OK\n";
}

void test_determinant_3x3() {
    std::cout << "[TEST] Determinant 3x3 ... ";

    Matrix m = {
        {6.0, 1.0, 1.0},
        {4.0, -2.0, 5.0},
        {2.0, 8.0, 7.0}
    };

    const double result = m.determinant();

    assert(std::abs(result + 306.0) < 1e-9);

    std::cout << "OK\n";
}

void test_determinant_non_square_error() {
    std::cout << "[TEST] Determinant non-square validation ... ";

    Matrix m = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    bool thrown = false;

    try {
        [[maybe_unused]] const double result =
            m.determinant();
    } catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);

    std::cout << "OK\n";
}

void test_determinant_empty_matrix_error() {
    std::cout << "[TEST] Determinant empty matrix validation ... ";

    Matrix m;

    bool thrown = false;

    try {
        [[maybe_unused]] const double result =
            m.determinant();
    } catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);

    std::cout << "OK\n";
}

void test_matrix_inverse() {
    std::cout << "[TEST] Matrix inverse ... ";

    Matrix m = {
        {4.0, 7.0},
        {2.0, 6.0}
    };

    const Matrix inverse = m.inverse();

    assert(std::abs(inverse(0, 0) - 0.6) < 1e-9);
    assert(std::abs(inverse(0, 1) + 0.7) < 1e-9);
    assert(std::abs(inverse(1, 0) + 0.2) < 1e-9);
    assert(std::abs(inverse(1, 1) - 0.4) < 1e-9);

    std::cout << "OK\n";
}

void test_matrix_inverse_errors() {
    std::cout << "[TEST] Matrix inverse validation ... ";

    bool non_square_thrown = false;

    try {
        Matrix non_square = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}
        };

        [[maybe_unused]] const Matrix result =
            non_square.inverse();
    } catch (const std::invalid_argument&) {
        non_square_thrown = true;
    }

    assert(non_square_thrown);

    bool singular_thrown = false;

    try {
        Matrix singular = {
            {1.0, 2.0},
            {2.0, 4.0}
        };

        [[maybe_unused]] const Matrix result =
            singular.inverse();
    } catch (const std::invalid_argument&) {
        singular_thrown = true;
    }

    assert(singular_thrown);

    std::cout << "OK\n";
}

} // namespace

int main() {
    std::cout << "Running matrix tests...\n\n";

    // Basic matrix operations
    test_matrix_construction();
    test_matrix_transpose();
    test_matrix_vector_multiplication();

    // Basic arithmetic operations
    test_matrix_addition();
    test_matrix_subtraction();
    test_matrix_multiplication();

    // In-place operations
    test_inplace_addition();
    test_inplace_subtraction();

    // Scalar operations
    test_scalar_operations();
    test_inplace_scalar_operations();

    // Matrix utilities
    test_factory_methods();
    test_row_operations();
    test_random_matrix();

    // Validation
    test_dimension_errors();
    test_scalar_division_by_zero();

    // Vector utilities
    test_euclidean_distance();
    test_euclidean_distance_dimension_error();

    // Identity matrix
    test_identity_matrix();
    test_identity_matrix_multiplication();

    // Determinant
    test_determinant_1x1();
    test_determinant_2x2();
    test_determinant_3x3();
    test_determinant_non_square_error();
    test_determinant_empty_matrix_error();

    // Matrix inverse
    test_matrix_inverse();
    test_matrix_inverse_errors();

    std::cout << "\nAll matrix tests passed!\n";

    return 0;
}