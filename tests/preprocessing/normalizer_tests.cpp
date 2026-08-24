#include "ml/preprocessing/normalizer.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace {

bool almost_equal(
    double a,
    double b,
    double epsilon = 1e-9
) {
    return std::abs(a - b) < epsilon;
}

void test_row_normalization() {
    ml::Matrix data{
        {3.0, 4.0},
        {5.0, 12.0}
    };

    ml::Normalizer normalizer;

    const ml::Matrix result =
        normalizer.transform(data);

    assert(almost_equal(result(0, 0), 0.6));
    assert(almost_equal(result(0, 1), 0.8));

    assert(almost_equal(result(1, 0), 5.0 / 13.0));
    assert(almost_equal(result(1, 1), 12.0 / 13.0));
}

void test_unit_norm() {
    ml::Matrix data{
        {3.0, 4.0},
        {5.0, 12.0}
    };

    ml::Normalizer normalizer;

    const ml::Matrix result =
        normalizer.transform(data);

    for (size_t row = 0; row < result.rows; ++row) {
        double squared_norm = 0.0;

        for (size_t column = 0; column < result.cols; ++column) {
            squared_norm +=
                result(row, column) *
                result(row, column);
        }

        assert(
            almost_equal(
                std::sqrt(squared_norm),
                1.0
            )
        );
    }
}
void test_zero_row() {
    ml::Matrix data{
        {0.0, 0.0},
        {3.0, 4.0}
    };

    ml::Normalizer normalizer;

    const ml::Matrix result =
        normalizer.transform(data);

    assert(almost_equal(result(0, 0), 0.0));
    assert(almost_equal(result(0, 1), 0.0));

    assert(almost_equal(result(1, 0), 0.6));
    assert(almost_equal(result(1, 1), 0.8));
}

void test_empty_input() {
    ml::Normalizer normalizer;

    bool threw = false;

    try {
        normalizer.transform(ml::Matrix{});
    } catch (const std::invalid_argument&) {
        threw = true;
    }

    assert(threw);
}

} // namespace

int main() {
    test_row_normalization();
    test_unit_norm();
    test_zero_row();
    test_empty_input();

    return 0;
}