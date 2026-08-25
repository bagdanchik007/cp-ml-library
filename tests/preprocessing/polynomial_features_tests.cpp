#include "ml/preprocessing/polynomial_features.hpp"

#include <cassert>
#include <stdexcept>

namespace {

void test_degree_two() {
    ml::Matrix data{
        {2.0},
        {3.0}
    };

    ml::PolynomialFeatures transformer(2);

    const ml::Matrix result =
        transformer.transform(data);

    assert(result.rows == 2);
    assert(result.cols == 2);

    assert(result(0, 0) == 2.0);
    assert(result(0, 1) == 4.0);

    assert(result(1, 0) == 3.0);
    assert(result(1, 1) == 9.0);
}

void test_degree_three() {
    ml::Matrix data{
        {2.0, 3.0}
    };

    ml::PolynomialFeatures transformer(3);

    const ml::Matrix result =
        transformer.transform(data);

    assert(result.rows == 1);
    assert(result.cols == 6);

    assert(result(0, 0) == 2.0);
    assert(result(0, 1) == 4.0);
    assert(result(0, 2) == 8.0);

    assert(result(0, 3) == 3.0);
    assert(result(0, 4) == 9.0);
    assert(result(0, 5) == 27.0);
}

void test_fit_transform() {
    ml::Matrix data{
        {2.0}
    };

    ml::PolynomialFeatures transformer(3);

    const ml::Matrix result =
        transformer.fit_transform(data);

    assert(result.rows == 1);
    assert(result.cols == 3);

    assert(result(0, 0) == 2.0);
    assert(result(0, 1) == 4.0);
    assert(result(0, 2) == 8.0);
}

void test_invalid_degree() {
    bool threw = false;

    try {
        ml::PolynomialFeatures transformer(0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

void test_empty_input() {
    ml::PolynomialFeatures transformer;

    bool threw = false;

    try {
        transformer.transform(ml::Matrix{});
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

} // namespace

int main() {
    test_degree_two();
    test_degree_three();
    test_fit_transform();
    test_invalid_degree();
    test_empty_input();

    return 0;
}