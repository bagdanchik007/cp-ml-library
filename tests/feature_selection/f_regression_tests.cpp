#include "ml/feature_selection/f_regression.hpp"
#include "ml/core/matrix/matrix.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void test_basic_f_regression()
{
    ml::Matrix X(5, 2);
    ml::Matrix y(5, 1);

    X(0, 0) = 1.0;
    X(1, 0) = 2.0;
    X(2, 0) = 3.0;
    X(3, 0) = 4.0;
    X(4, 0) = 5.0;

    X(0, 1) = 5.0;
    X(1, 1) = 4.0;
    X(2, 1) = 3.0;
    X(3, 1) = 2.0;
    X(4, 1) = 1.0;

    y(0, 0) = 1.0;
    y(1, 0) = 2.0;
    y(2, 0) = 3.0;
    y(3, 0) = 4.0;
    y(4, 0) = 5.0;

    const auto result = ml::f_regression(X, y);

    assert(result.scores.size() == 2);
    assert(result.p_values.size() == 2);

    assert(std::isinf(result.scores[0]));
    assert(std::isinf(result.scores[1]));

    assert(result.p_values[0] == 0.0);
    assert(result.p_values[1] == 0.0);
}

void test_constant_feature()
{
    ml::Matrix X(4, 1);
    ml::Matrix y(4, 1);

    X(0, 0) = 5.0;
    X(1, 0) = 5.0;
    X(2, 0) = 5.0;
    X(3, 0) = 5.0;

    y(0, 0) = 1.0;
    y(1, 0) = 2.0;
    y(2, 0) = 3.0;
    y(3, 0) = 4.0;

    const auto result = ml::f_regression(X, y);

    assert(result.scores.size() == 1);
    assert(result.scores[0] == 0.0);
    assert(result.p_values[0] == 1.0);
}

void test_invalid_dimensions()
{
    ml::Matrix X(3, 2);
    ml::Matrix y(4, 1);

    bool thrown = false;

    try {
        ml::f_regression(X, y);
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void test_invalid_y_columns()
{
    ml::Matrix X(3, 2);
    ml::Matrix y(3, 2);

    bool thrown = false;

    try {
        ml::f_regression(X, y);
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

void test_too_few_samples()
{
    ml::Matrix X(2, 1);
    ml::Matrix y(2, 1);

    bool thrown = false;

    try {
        ml::f_regression(X, y);
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }

    assert(thrown);
}

} // namespace

int main()
{
    test_basic_f_regression();
    test_constant_feature();
    test_invalid_dimensions();
    test_invalid_y_columns();
    test_too_few_samples();

    std::cout << "All F-regression tests passed.\n";

    return 0;
}