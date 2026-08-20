#include "ml/ml.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ml;

namespace {

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

void test_matrix_index_validation() {
    std::cout << "[TEST] Matrix index validation ... ";

    Matrix m = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    bool row_thrown = false;

    try {
        [[maybe_unused]] const double value = m(2, 0);
    } catch (const std::out_of_range&) {
        row_thrown = true;
    }

    assert(row_thrown);

    bool column_thrown = false;

    try {
        [[maybe_unused]] const double value = m(0, 2);
    } catch (const std::out_of_range&) {
        column_thrown = true;
    }

    assert(column_thrown);

    std::cout << "OK\n";
}

void test_matrix_row_validation() {
    std::cout << "[TEST] Matrix row validation ... ";

    Matrix m = {
        {1.0, 2.0},
        {3.0, 4.0}
    };

    bool row_access_thrown = false;

    try {
        [[maybe_unused]] const auto result = m.row(2);
    } catch (const std::out_of_range&) {
        row_access_thrown = true;
    }

    assert(row_access_thrown);

    bool row_set_thrown = false;

    try {
        m.set_row(2, {10.0, 20.0});
    } catch (const std::out_of_range&) {
        row_set_thrown = true;
    }

    assert(row_set_thrown);

    std::cout << "OK\n";
}

void test_matrix_inverse_validation() {
    std::cout << "[TEST] Matrix inverse validation ... ";

    const Matrix non_square = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    bool non_square_thrown = false;

    try {
        [[maybe_unused]] const Matrix result =
            non_square.inverse();
    } catch (const std::invalid_argument&) {
        non_square_thrown = true;
    }

    assert(non_square_thrown);

    const Matrix singular = {
        {1.0, 2.0},
        {2.0, 4.0}
    };

    bool singular_thrown = false;

    try {
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
    std::cout << "Running matrix validation tests...\n\n";

    test_dimension_errors();
    test_scalar_division_by_zero();
    test_matrix_index_validation();
    test_matrix_row_validation();
    test_matrix_inverse_validation();
    std::cout << "\nAll matrix validation tests passed!\n";

    return 0;
}