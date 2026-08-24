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

void test_normalize_rows() {
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
        {3.0, 4.0}
    };

    ml::Normalizer normalizer;

    const ml::Matrix result =
        normalizer.transform(data);

    const double norm =
        std::sqrt(
            result(0, 0) * result(0, 0) +
            result(0, 1) * result(0, 1)
        );

    assert(almost_equal(norm, 1.0));
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
    test_normalize_rows();
    test_unit_norm();
    test_zero_row();
    test_empty_input();

    return 0;
}