#include "ml/ml.hpp"

#include <cassert>
#include <iostream>

using namespace ml;

namespace {

void test_matrix_construction() {
    std::cout << "[TEST] Matrix construction ... ";

    Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    assert(matrix.rows == 2);
    assert(matrix.cols == 2);
    assert(matrix.data.size() == 4);

    assert(matrix(0, 0) == 1.0);
    assert(matrix(0, 1) == 2.0);
    assert(matrix(1, 0) == 3.0);
    assert(matrix(1, 1) == 4.0);

    std::cout << "OK\n";
}

void test_matrix_element_access() {
    std::cout << "[TEST] Matrix element access ... ";

    Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    assert(matrix(0, 0) == 1.0);
    assert(matrix(0, 1) == 2.0);
    assert(matrix(1, 0) == 3.0);
    assert(matrix(1, 1) == 4.0);

    matrix(0, 0) = 10.0;
    matrix(1, 1) = 20.0;

    assert(matrix(0, 0) == 10.0);
    assert(matrix(1, 1) == 20.0);

    std::cout << "OK\n";
}

void test_matrix_transpose() {
    std::cout << "[TEST] Matrix transpose ... ";

    const Matrix matrix = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    const Matrix transposed = matrix.transpose();

    assert(transposed.rows == 3);
    assert(transposed.cols == 2);

    assert(transposed(0, 0) == 1.0);
    assert(transposed(0, 1) == 4.0);
    assert(transposed(1, 0) == 2.0);
    assert(transposed(1, 1) == 5.0);
    assert(transposed(2, 0) == 3.0);
    assert(transposed(2, 1) == 6.0);

    std::cout << "OK\n";
}

void test_matrix_vector_multiplication() {
    std::cout << "[TEST] Matrix-vector multiplication ... ";

    const Matrix matrix = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    const std::vector<double> vector = {
        5.0,
        6.0
    };

    const auto result = matrix * vector;

    assert(result.size() == 2);
    assert(result[0] == 17.0);
    assert(result[1] == 39.0);

    std::cout << "OK\n";
}

void test_column_operations() {
    std::cout << "[TEST] Column operations ... ";

    Matrix m = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    const auto column = m.column(1);

    assert(column.size() == 3);
    assert(column[0] == 2.0);
    assert(column[1] == 5.0);
    assert(column[2] == 8.0);

    std::cout << "OK\n";
}

} // namespace

int main() {
    std::cout << "Running matrix tests...\n\n";

    test_matrix_construction();
    test_matrix_element_access();
    test_matrix_transpose();
    test_matrix_vector_multiplication();
    test_column_operations();

    std::cout << "\nAll matrix tests passed!\n";

    return 0;
}